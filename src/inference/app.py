"""ML Inference Service + Decision Engine + Case Management API.

One FastAPI app (microservice) that exposes:
  GET  /health                      - liveness + model status
  POST /v1/transactions             - validate and enqueue a live event
  POST /v1/score                    - model scoring + business rules
  GET  /v1/cases                    - open fraud cases (analyst queue)
  POST /v1/cases/{id}/verdict       - analyst feedback -> retraining labels

The decision engine combines the model probability with BUSINESS RULES
(velocity override) - pure model scores are never shipped to production
without a rules layer on top.
"""
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.common.config import settings
from src.common.features import FEATURES
from src.common.schemas import TOPIC_RAW, Decision, Transaction
from src.inference.model_loader import load_model

STATE = {"bundle": None}
_kafka_producer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _kafka_producer
    try:
        STATE["bundle"] = load_model()
    except Exception as e:
        print(f"[warn] starting WITHOUT model: {e}")
        STATE["bundle"] = None
    yield
    if _kafka_producer is not None:
        _kafka_producer.flush(timeout=5)
        _kafka_producer.close(timeout=5)
        _kafka_producer = None
    STATE["bundle"] = None


app = FastAPI(title="Fraud Detection - ML Inference Service",
              version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------- DB layer
_engine = None


def get_engine():
    """Lazy SQLAlchemy engine - API stays healthy even if Postgres is down."""
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine
        _engine = create_engine(settings.postgres_dsn, pool_pre_ping=True)
    return _engine


def _db_ready() -> bool:
    try:
        from sqlalchemy import text
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def insert_case(txn: Transaction, p_fraud: float, decision: str) -> Optional[int]:
    try:
        from sqlalchemy import text
        with get_engine().begin() as conn:
            row = conn.execute(text("""
                INSERT INTO cases (txn_id, decision, p_fraud, amount, user_id)
                VALUES (:txn_id, :decision, :p_fraud, :amount, :user_id)
                ON CONFLICT (txn_id) DO NOTHING
                RETURNING case_id
            """), {"txn_id": txn.txn_id, "decision": decision,
                   "p_fraud": p_fraud, "amount": txn.amount,
                   "user_id": txn.user_id}).fetchone()
            if row:
                return int(row[0])
            row = conn.execute(text(
                "SELECT case_id FROM cases WHERE txn_id = :t"),
                {"t": txn.txn_id}).fetchone()
            return int(row[0]) if row else None
    except Exception as e:
        print(f"[warn] case insert failed ({e.__class__.__name__}: "
              f"{str(e)[:120]})")
        return None


def persist_scored(txn: Transaction, p_fraud: float, decision: str,
                   model_version: str):
    try:
        from sqlalchemy import text
        with get_engine().begin() as conn:
            conn.execute(text("""
                INSERT INTO scored_transactions
                    (txn_id, event_ts, user_id, card_id, merchant_id,
                     amount, p_fraud, decision, model_version)
                VALUES (:txn_id, :event_ts, :user_id, :card_id, :merchant_id,
                        :amount, :p_fraud, :decision, :model_version)
                ON CONFLICT (txn_id) DO NOTHING
            """), {"txn_id": txn.txn_id, "event_ts": txn.event_ts,
                   "user_id": txn.user_id, "card_id": txn.card_id,
                   "merchant_id": txn.merchant_id, "amount": txn.amount,
                   "p_fraud": p_fraud, "decision": decision,
                   "model_version": model_version})
    except Exception as e:
        print(f"[warn] scored_transactions insert failed "
              f"({e.__class__.__name__})")


# ------------------------------------------------------------- contracts
class ScoreRequest(BaseModel):
    transaction: Transaction
    features: dict


class ScoreResponse(BaseModel):
    txn_id: str
    p_fraud: float
    decision: Decision
    reason: str
    case_id: Optional[int] = None
    model_version: str


class VerdictRequest(BaseModel):
    verdict: str  # "FRAUD" | "LEGIT"


class TransactionAccepted(BaseModel):
    txn_id: str
    status: str
    topic: str
    partition: int
    offset: int


# ------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": STATE["bundle"] is not None,
        "model_version": STATE["bundle"].version if STATE["bundle"] else None,
        "db_ready": _db_ready(),
        "features": FEATURES,
    }


@app.post("/v1/transactions", response_model=TransactionAccepted,
          status_code=202)
def ingest_transaction(txn: Transaction):
    """Validate and enqueue a new transaction for live stream scoring."""
    global _kafka_producer
    try:
        if _kafka_producer is None:
            from kafka import KafkaProducer

            _kafka_producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap,
                key_serializer=lambda value: value.encode("utf-8"),
                value_serializer=lambda value: json.dumps(
                    value, default=str
                ).encode("utf-8"),
                linger_ms=5,
            )
        metadata = _kafka_producer.send(
            TOPIC_RAW,
            key=txn.card_id,
            value=txn.model_dump(mode="json"),
        ).get(timeout=10)
    except Exception as e:
        print(f"[warn] transaction enqueue failed ({e.__class__.__name__})")
        raise HTTPException(
            status_code=503,
            detail=f"transaction could not be queued: {e.__class__.__name__}",
        ) from e

    return TransactionAccepted(
        txn_id=txn.txn_id,
        status="queued",
        topic=metadata.topic,
        partition=metadata.partition,
        offset=metadata.offset,
    )


def decide(p_fraud: float, features: dict) -> tuple:
    """Decision engine: model score + business rules."""
    if p_fraud >= settings.block_threshold:
        return Decision.BLOCK, f"score {p_fraud:.3f} >= block {settings.block_threshold}"
    if p_fraud >= settings.review_threshold:
        return Decision.REVIEW, f"score {p_fraud:.3f} >= review {settings.review_threshold}"

    velocity = float(features.get("card_txn_count_10m", 0) or 0)
    if velocity >= settings.velocity_max_10min:
        return Decision.REVIEW, (f"velocity rule: {int(velocity)} txns/10min "
                                 f">= {settings.velocity_max_10min}")
    return Decision.APPROVE, "normal"


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    bundle = STATE["bundle"]
    if bundle is None:
        # fail-safe: never approve blindly when the model is down
        return JSONResponse(status_code=503, content={
            "detail": "model not loaded - run `make train`"})

    missing = [f for f in FEATURES if f not in req.features]
    if missing:
        raise HTTPException(status_code=422,
                            detail=f"missing features: {missing}")

    # DEFECT-03 hardening: the model must only ever see finite numbers.
    # Strings / NaN / Inf from a broken upstream are contract violations
    # (422), never a 500 and never a silent APPROVE.
    import math

    clean = {}
    for f in FEATURES:
        try:
            v = float(req.features[f])
        except (TypeError, ValueError):
            raise HTTPException(status_code=422,
                                detail=f"non-numeric feature: {f}")
        if not math.isfinite(v):
            raise HTTPException(status_code=422,
                                detail=f"non-finite feature: {f}")
        clean[f] = v

    from src.common.features import to_vector
    import pandas as pd

    X = pd.DataFrame([to_vector(clean)], columns=FEATURES)
    try:
        p_fraud = float(bundle.model.predict_proba(X)[0][1])
    except Exception as e:  # fail-safe: scorer failure must not approve
        print(f"[warn] predict failed ({e.__class__.__name__})")
        return JSONResponse(status_code=503,
                            content={"detail": "scoring failed - fail-safe"})
    if not math.isfinite(p_fraud):
        return JSONResponse(status_code=503,
                            content={"detail": "non-finite score - fail-safe"})
    decision, reason = decide(p_fraud, clean)

    case_id = None
    if decision != Decision.APPROVE:
        case_id = insert_case(req.transaction, p_fraud, decision.value)
    persist_scored(req.transaction, p_fraud, decision.value, bundle.version)

    return ScoreResponse(
        txn_id=req.transaction.txn_id,
        p_fraud=round(p_fraud, 4),
        decision=decision,
        reason=reason,
        case_id=case_id,
        model_version=bundle.version,
    )


@app.get("/v1/cases")
def list_cases(status: str = Query("OPEN"), limit: int = Query(50, le=500)):
    if not _db_ready():
        return {"cases": [], "db_ready": False}
    from sqlalchemy import text
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT case_id, txn_id, status, decision, p_fraud, amount,
                   user_id, created_ts, analyst_verdict
            FROM cases WHERE status = :s
            ORDER BY created_ts DESC LIMIT :l
        """), {"s": status, "l": limit}).mappings().all()
    return {"cases": [dict(r) for r in rows], "db_ready": True}


@app.post("/v1/cases/{case_id}/verdict")
def case_verdict(case_id: int, req: VerdictRequest):
    if req.verdict not in ("FRAUD", "LEGIT"):
        raise HTTPException(400, "verdict must be FRAUD or LEGIT")
    if not _db_ready():
        raise HTTPException(503, "database unavailable")
    from sqlalchemy import text
    with get_engine().begin() as conn:
        row = conn.execute(text(
            "SELECT txn_id FROM cases WHERE case_id = :c"), {"c": case_id}
        ).fetchone()
        if not row:
            raise HTTPException(404, f"case {case_id} not found")
        txn_id = row[0]
        conn.execute(text("""
            UPDATE cases SET status='RESOLVED', resolved_ts=now(),
                              analyst_verdict=:v
            WHERE case_id=:c
        """), {"v": req.verdict, "c": case_id})
        conn.execute(text("""
            INSERT INTO analyst_labels (txn_id, true_label)
            VALUES (:t, :y)
        """), {"t": txn_id, "y": 1 if req.verdict == "FRAUD" else 0})
    return {"case_id": case_id, "status": "RESOLVED", "verdict": req.verdict,
            "label_saved": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
