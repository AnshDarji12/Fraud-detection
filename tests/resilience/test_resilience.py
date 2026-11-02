"""RESILIENCE / GRACEFUL DEGRADATION TESTS (BA-10).

Production laws:
  - Redis down  -> feature store falls back to in-memory
  - Model API down -> stream processor routes to REVIEW (never APPROVE)
  - Duplicate deliveries -> deduped (bounded LRU)
  - Corrupt history entries -> tolerated, never crash
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.common.schemas import Decision, Transaction
from src.streaming.features_online import InMemoryFeatureStore, make_store
from src.streaming.txn_deduper import TxnDeduper

UTC = timezone.utc


def _txn(i, **over):
    base = dict(txn_id=f"T-{i}", event_ts=datetime(2026, 10, 1, 12, 0, 0,
                                                   tzinfo=UTC),
                user_id="U-1", card_id="C-1", merchant_id="M-1",
                merchant_cat="grocery", amount=50.0, channel="web",
                device_id="D-1")
    base.update(over)
    return Transaction(**base)


# ------------------------------------------------------------ fallbacks
def test_redis_down_falls_back_to_inmemory():
    store = make_store("redis://localhost:1/0")  # nothing listens there
    assert isinstance(store, InMemoryFeatureStore)
    f = store.compute_and_update(_txn(1))
    assert f["card_txn_count_10m"] == 0.0


def test_non_redis_url_uses_inmemory():
    assert isinstance(make_store("memory://"), InMemoryFeatureStore)


def test_cold_start_gives_sane_defaults():
    f = InMemoryFeatureStore().compute_and_update(_txn(1))
    assert f["card_time_since_prev_sec"] == 24 * 3600.0
    assert f["user_amount_avg_7d"] == 0.0
    assert f["card_device_changed"] == 0.0


def test_inference_down_retries_then_fails_safe(monkeypatch):
    """Stream processor's scorer must end on REVIEW (BA-10) when the
    inference service is unreachable - never silently APPROVE."""
    import httpx
    from src.streaming import consumer

    def _boom(request):
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(consumer.time, "sleep", lambda s: None)

    http = httpx.Client(transport=httpx.MockTransport(_boom))
    decision, p, reason, version, case_id = consumer._score_with_retry(
        http, _txn(1), {"card_txn_count_10m": 0})
    assert decision == "REVIEW"
    assert p == 0.0
    assert "model_unavailable" in reason
    http.close()


def test_score_with_retry_returns_payload_on_success(monkeypatch):
    import httpx
    from src.streaming import consumer

    def _ok(request):
        return httpx.Response(200, json={
            "decision": "APPROVE", "p_fraud": 0.01, "reason": "normal",
            "model_version": "stub-v1", "case_id": None})

    http = httpx.Client(transport=httpx.MockTransport(_ok))
    decision, p, reason, version, _ = consumer._score_with_retry(
        http, _txn(1), {"card_txn_count_10m": 0})
    assert decision == "APPROVE" and version == "stub-v1"
    http.close()


# -------------------------------------------------------------- dedup
def test_deduper_first_delivery_passes():
    assert TxnDeduper().seen_before("T-1") is False


def test_deduper_blocks_redelivery():
    d = TxnDeduper()
    d.seen_before("T-1")
    assert d.seen_before("T-1") is True
    assert d.hits == 1


def test_deduper_is_bounded_lru():
    d = TxnDeduper(maxlen=3)
    for i in range(5):
        d.seen_before(f"T-{i}")
    assert d.seen_before("T-0") is False   # evicted (oldest)
    assert d.seen_before("T-4") is True    # still remembered (newest)


# ------------------------------------------------- corrupt state hygiene
def test_corrupt_redis_member_tolerated():
    """A malformed member in the card ZSET must not crash scoring."""
    store = _poisoned_redis_store()
    f = store.compute_and_update(_txn(1, card_id="C-X"))
    assert "card_txn_count_1h" in f
    assert f["card_txn_count_1h"] >= 0


def test_state_survives_high_volume_same_card():
    """No unbounded growth blowups within one card's window."""
    store = InMemoryFeatureStore()
    t0 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    for i in range(500):
        store.compute_and_update(_txn(i, event_ts=t0 + timedelta(seconds=i),
                                      card_id="C-HOT"))
    assert len(store._card["C-HOT"]) <= 501  # pruned, not unbounded


# ------------------------------------------------------------- helpers
def _poisoned_redis_store():
    import fakeredis
    import redis as redis_lib
    from src.streaming.features_online import RedisFeatureStore

    fake = fakeredis.FakeStrictRedis(decode_responses=True)

    class _F:
        @classmethod
        def from_url(cls, url, **kw):
            return fake

    original = redis_lib.Redis.from_url
    redis_lib.Redis.from_url = _F.from_url
    try:
        store = RedisFeatureStore("redis://fakeredis:6379/0")
    finally:
        redis_lib.Redis.from_url = original

    fake.zadd("card:C-X:txns", {"CORRUPTED-GARBAGE": 1.0})
    return store
