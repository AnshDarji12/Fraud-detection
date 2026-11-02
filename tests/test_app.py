"""API tests - model is replaced by a stub, DB stays offline."""
from fastapi.testclient import TestClient

import src.inference.app as app_module
from src.inference.app import app


class _Stub:
    """Returns a constant p_fraud - set per test."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return [[1 - self.p, self.p]]


def client_with(p: float) -> TestClient:
    app_module.STATE["bundle"] = type("B", (), {"model": _Stub(p),
                                                "version": "stub"})()
    return TestClient(app)


FEATURES_OK = {
    "amount": 100.0, "log_amount": 4.6, "hour_of_day": 12, "is_night": 0,
    "card_txn_count_10m": 1, "card_txn_count_1h": 1, "card_amount_sum_1h": 100,
    "user_txn_count_24h": 3, "user_amount_avg_7d": 80.0,
    "card_time_since_prev_sec": 300, "card_device_changed": 0,
    "card_merchant_changed": 0,
}

TXN = {
    "txn_id": "T-TEST-1", "event_ts": "2026-10-01T12:00:00Z",
    "user_id": "U-1", "card_id": "C-1", "merchant_id": "M-1",
    "merchant_cat": "grocery", "amount": 100.0, "currency": "USD",
    "channel": "web", "device_id": "D-1", "ip": "10.0.0.1",
    "lat": None, "lon": None,
}


def test_health_ok():
    c = client_with(0.1)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_transaction_ingest_publishes_to_raw_topic():
    class _Future:
        def get(self, timeout):
            return type("Metadata", (), {
                "topic": "txn.raw", "partition": 0, "offset": 17,
            })()

    class _Producer:
        def send(self, topic, key, value):
            assert topic == "txn.raw"
            assert key == TXN["card_id"]
            assert value["txn_id"] == TXN["txn_id"]
            return _Future()

    app_module._kafka_producer = _Producer()
    try:
        response = client_with(0.1).post("/v1/transactions", json=TXN)
    finally:
        app_module._kafka_producer = None

    assert response.status_code == 202
    assert response.json() == {
        "txn_id": TXN["txn_id"],
        "status": "queued",
        "topic": "txn.raw",
        "partition": 0,
        "offset": 17,
    }


def test_score_block_path():
    c = client_with(0.93)
    r = c.post("/v1/score", json={"transaction": TXN, "features": FEATURES_OK})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "BLOCK"
    assert body["p_fraud"] >= 0.85


def test_score_velocity_override():
    feats = dict(FEATURES_OK, card_txn_count_10m=9)
    c = client_with(0.05)  # model says fine...
    r = c.post("/v1/score", json={"transaction": TXN, "features": feats})
    body = r.json()
    # ...but the business rule forces REVIEW
    assert body["decision"] == "REVIEW"
    assert "velocity" in body["reason"]


def test_score_missing_feature_422():
    c = client_with(0.1)
    bad = dict(FEATURES_OK)
    bad.pop("card_txn_count_10m")
    r = c.post("/v1/score", json={"transaction": TXN, "features": bad})
    assert r.status_code == 422
    assert "card_txn_count_10m" in r.json()["detail"]


def test_verdict_validation():
    c = client_with(0.1)
    r = c.post("/v1/cases/1/verdict", json={"verdict": "MAYBE"})
    assert r.status_code == 400
