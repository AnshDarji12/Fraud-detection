"""Stream Processor - the heart of the real-time pipeline.

Consumes `txn.raw`:
  1. computes sliding-window behavioural features (Redis feature store)
  2. calls the ML Inference Service  POST /v1/score
  3. publishes decision to `txn.scored`
  4. non-APPROVE decisions are also published to `txn.alerts`
  5. any failure is sent to the dead-letter topic `txn.dlq` (enterprise
     hygiene: a poison message must never kill the pipeline)

NOTE: this Python consumer keeps the laptop demo light. The same
feature+score logic can be lifted into a Spark Structured Streaming /
Flink job for horizontal scale (see README - Roadmap).
"""
import json
import time

import httpx

from src.common.config import settings
from src.common.schemas import (TOPIC_ALERTS, TOPIC_DLQ, TOPIC_RAW,
                                TOPIC_SCORED, Transaction)
from src.streaming.features_online import make_store
from src.streaming.txn_deduper import TxnDeduper

try:
    from kafka import KafkaConsumer, KafkaProducer
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install kafka-python-ng  (see requirements.txt)") from e


def json_serializer(v) -> bytes:
    return json.dumps(v, default=str).encode("utf-8")


def main():
    print(f"[..] connecting to Kafka at {settings.kafka_bootstrap}")
    consumer = KafkaConsumer(
        TOPIC_RAW,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id="fraud-processor",
        auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        consumer_timeout_ms=1000,
    )
    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        value_serializer=json_serializer,
        linger_ms=5,
    )
    store = make_store(settings.redis_url)
    deduper = TxnDeduper()          # DEFECT-02: at-least-once -> score once
    http = httpx.Client(timeout=5.0)
    print(f"[ok] stream processor running - consuming {TOPIC_RAW}")

    scored_count, alert_count, dup_count = 0, 0, 0
    try:
        while True:
            batch = consumer.poll(timeout_ms=1000)
            if not batch:
                continue
            for _tp, messages in batch.items():
                for msg in messages:
                    try:
                        txn = Transaction(**msg.value)
                    except Exception as e:
                        print(f"[dlq] bad payload: {e}")
                        producer.send(TOPIC_DLQ, {"error": str(e),
                                                  "payload": msg.value})
                        continue

                    if deduper.seen_before(txn.txn_id):
                        dup_count += 1
                        continue

                    features = store.compute_and_update(txn)

                    decision, p_fraud, reason, model_version, case_id = \
                        _score_with_retry(http, txn, features)

                    scored = txn.model_dump(mode="json")
                    scored.update({
                        "features": features,
                        "p_fraud": p_fraud,
                        "decision": decision,
                        "reason": reason,
                        "model_version": model_version,
                        "case_id": case_id,
                    })
                    producer.send(TOPIC_SCORED, scored)
                    scored_count += 1

                    if decision != "APPROVE":
                        producer.send(TOPIC_ALERTS, scored)
                        alert_count += 1

                    if scored_count % 100 == 0:
                        producer.flush()
                        print(f"[..] scored={scored_count} alerts={alert_count} "
                              f"dupes={dup_count} "
                              f"last={txn.txn_id} p={p_fraud:.3f} -> {decision}")
    except KeyboardInterrupt:
        print("\n[ok] shutting down stream processor")
    finally:
        producer.flush()
        producer.close()
        consumer.close()
        http.close()


def _score_with_retry(http: httpx.Client, txn: Transaction, features: dict,
                      retries: int = 3):
    """Call the inference service; degrade gracefully if it is down."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = http.post(f"{settings.inference_url}/v1/score", json={
                "transaction": txn.model_dump(mode="json"),
                "features": features,
            })
            resp.raise_for_status()
            body = resp.json()
            return (body.get("decision", "REVIEW"),
                    float(body.get("p_fraud", 0.0)),
                    body.get("reason", ""),
                    body.get("model_version", "unknown"),
                    body.get("case_id"))
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    print(f"[warn] inference unavailable after {retries} tries: {last_err}")
    return "REVIEW", 0.0, f"model_unavailable: {last_err.__class__.__name__}", \
        "none", None


if __name__ == "__main__":
    main()
