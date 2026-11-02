"""API <-> REAL MODEL integration tests.

Unlike tests/test_app.py (stubbed model), these exercise the REAL
trained artifact through the real HTTP layer.
"""
import pytest
from fastapi.testclient import TestClient

import src.inference.app as app_module
from src.common.features import FEATURES, to_vector
from src.inference.app import app
from src.streaming.features_online import InMemoryFeatureStore


@pytest.fixture(scope="module")
def client(bundle):
    app_module.STATE["bundle"] = bundle
    return TestClient(app)


def _txn_payload(row):
    import pandas as pd
    return {
        "txn_id": row.txn_id, "event_ts": row.event_ts.isoformat(),
        "user_id": row.user_id, "card_id": row.card_id,
        "merchant_id": row.merchant_id, "merchant_cat": row.merchant_cat,
        "amount": float(row.amount), "currency": "USD",
        "channel": row.channel, "device_id": row.device_id, "ip": row.ip,
        "lat": float(row.lat) if pd.notna(row.lat) else None,
        "lon": float(row.lon) if pd.notna(row.lon) else None,
    }


def test_health_with_real_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "local"
    assert body["features"] == FEATURES


def test_score_real_event_end_to_end(client, test_split):
    """CSV row -> online features -> HTTP scoring -> valid verdict."""
    store = InMemoryFeatureStore()
    for row in test_split.iloc[:5].itertuples(index=False):
        from src.common.schemas import Transaction
        t = Transaction(**{k: v for k, v in _txn_payload(row).items()})
        f = store.compute_and_update(t)
        r = client.post("/v1/score", json={"transaction": _txn_payload(row),
                                           "features": f})
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["p_fraud"] <= 1.0
        assert body["decision"] in ("APPROVE", "REVIEW", "BLOCK")
        assert body["model_version"] == "local"
        assert body["reason"]


def test_score_contract_enforced_over_http(client, test_split):
    row = next(test_split.iloc[:5].itertuples(index=False))
    payload = _txn_payload(row)
    del payload["card_id"]
    r = client.post("/v1/score", json={
        "transaction": payload, "features": {k: 0 for k in FEATURES}})
    assert r.status_code == 422


def test_score_missing_features_422(client, test_split):
    row = next(test_split.iloc[:5].itertuples(index=False))
    f = {k: 0 for k in FEATURES}
    del f["card_txn_count_10m"]
    r = client.post("/v1/score", json={"transaction": _txn_payload(row),
                                       "features": f})
    assert r.status_code == 422


def test_cases_endpoint_graceful_without_db(client):
    r = client.get("/v1/cases")
    assert r.status_code == 200
    assert r.json()["db_ready"] is False


def test_verdict_rejects_unknown_value(client):
    r = client.post("/v1/cases/1/verdict", json={"verdict": "MAYBE"})
    assert r.status_code == 400
