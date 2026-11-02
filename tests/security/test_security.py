"""SECURITY TESTS - hostile input, fail-safe behavior, info hygiene.

The scoring API is internet-facing (checkout path). Everything here is
free/open-source: FastAPI TestClient + pydantic contract testing.
"""
import math

import pytest
from fastapi.testclient import TestClient

import src.inference.app as app_module
from src.common.features import FEATURES
from src.inference.app import app

FEATURES_OK = {
    "amount": 100.0, "log_amount": 4.6, "hour_of_day": 12, "is_night": 0,
    "card_txn_count_10m": 1, "card_txn_count_1h": 1,
    "card_amount_sum_1h": 100, "user_txn_count_24h": 3,
    "user_amount_avg_7d": 80.0, "card_time_since_prev_sec": 300,
    "card_device_changed": 0, "card_merchant_changed": 0,
}

TXN = {
    "txn_id": "T-SEC-1", "event_ts": "2026-10-01T12:00:00Z",
    "user_id": "U-1", "card_id": "C-1", "merchant_id": "M-1",
    "merchant_cat": "grocery", "amount": 100.0, "currency": "USD",
    "channel": "web", "device_id": "D-1", "ip": "10.0.0.1",
    "lat": None, "lon": None,
}


class _Stub:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return [[1 - self.p, self.p]]


@pytest.fixture()
def client():
    app_module.STATE["bundle"] = type(
        "B", (), {"model": _Stub(0.10), "version": "stub"})()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _score(client, features=None, txn=None):
    return client.post("/v1/score", json={
        "transaction": txn or TXN, "features": features or FEATURES_OK})


# ------------------------------------------------------- input hardening
def test_non_finite_rejected_with_422(client):
    """Contract: features must be finite numbers - NaN/Inf are rejected
    at the API edge (float('NaN') parses them, so the server must check)."""
    for bad in ("NaN", "Infinity", "-Infinity"):
        f = dict(FEATURES_OK, card_txn_count_10m=bad)
        r = _score(client, features=f)
        assert r.status_code == 422, (bad, r.status_code)
        assert "non-finite" in r.json()["detail"]


def test_non_numeric_feature_rejected_422(client):
    f = dict(FEATURES_OK, amount="lots")
    r = _score(client, features=f)
    assert r.status_code == 422
    assert "non-numeric" in r.json()["detail"]


def test_nan_feature_never_500(client):
    """NaN arrives as the JSON literal 'NaN' from broken upstreams
    (JSON has no real NaN); the API must reject it, never 500."""
    f = dict(FEATURES_OK, card_txn_count_10m="NaN")
    r = _score(client, features=f)
    assert r.status_code in (200, 422), r.status_code
    if r.status_code == 200:
        assert r.json()["decision"] in ("REVIEW", "BLOCK"), (
            "non-finite input must degrade to fail-safe, not APPROVE")


def test_inf_feature_never_500(client):
    f = dict(FEATURES_OK, card_time_since_prev_sec="Infinity")
    r = _score(client, features=f)
    assert r.status_code in (200, 422), r.status_code
    if r.status_code == 200:
        assert r.json()["decision"] in ("REVIEW", "BLOCK")


def test_sql_injection_strings_are_inert(client):
    t = dict(TXN, user_id="'; DROP TABLE cases; --",
             merchant_id="1' OR '1'='1")
    r = _score(client, txn=t)
    assert r.status_code == 200
    body = r.json()
    assert body["txn_id"] == t["txn_id"]  # echoed verbatim, never executed


def test_xss_payload_is_inert_json(client):
    t = dict(TXN, txn_id="<script>alert(1)</script>")
    r = _score(client, txn=t)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")


def test_path_traversal_string_inert(client):
    t = dict(TXN, device_id="../../etc/passwd")
    assert _score(client, txn=t).status_code == 200


def test_extreme_amount_handled(client):
    r = _score(client, txn=dict(TXN, amount=1e9))
    assert r.status_code in (200, 422)


def test_negative_amount_rejected_at_http(client):
    r = _score(client, txn=dict(TXN, amount=-500))
    assert r.status_code == 422


# ------------------------------------------------------- fail-safe logic
def test_model_down_returns_503_never_approve(client):
    app_module.STATE["bundle"] = None
    r = _score(client)
    assert r.status_code == 503  # BA-10: fail-safe, never blind-approve


def test_health_endpoint_no_internal_leak(client):
    app_module.STATE["bundle"] = None
    r = client.get("/health")
    body = r.text.lower()
    for secret in ("password", "postgres_dsn", "kafka_bootstrap", "dsn"):
        assert secret not in body, f"health leaks {secret}"


def test_error_responses_do_not_contain_stacktrace(client):
    app_module.STATE["bundle"] = None
    r = _score(client)
    assert "traceback" not in r.text.lower()
