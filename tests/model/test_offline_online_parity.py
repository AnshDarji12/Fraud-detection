"""OFFLINE vs ONLINE FEATURE PARITY - the classic fraud-ML production bug.

The model is trained on offline features (src/training/features.py) but
served with online features (src/streaming/features_online.py). If they
ever diverge, training/serving skew silently degrades the model.

This suite pins parity across THREE backends:
  offline (pandas)  <->  InMemoryFeatureStore  <->  RedisFeatureStore
                                                (fakeredis, no Docker)
including boundary semantics: exact window edges, same-timestamp events,
and cards that go quiet for over a week.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.common.features import FEATURES
from src.common.schemas import Transaction
from src.streaming.features_online import (InMemoryFeatureStore,
                                           RedisFeatureStore)

T0 = datetime(2026, 10, 1, 12, 0, 0)


def txn(i, ts, card="C-1", user="U-1", amount=50.0, device="D-1",
        merchant="M-1"):
    return Transaction(txn_id=f"T-{i:05d}", event_ts=ts, user_id=user,
                       card_id=card, merchant_id=merchant,
                       merchant_cat="grocery", amount=amount,
                       channel="web", device_id=device)


def offline_features(txns) -> pd.DataFrame:
    from src.training.features import build_features
    df = pd.DataFrame([{**t.model_dump(), "is_fraud": 0} for t in txns])
    return build_features(df)


def assert_parity(offline: pd.DataFrame, store, txns):
    assert len(offline) == len(txns)
    for row, t in zip(offline.itertuples(index=False), txns):
        f = store.compute_and_update(t)
        for name in FEATURES:
            expected = getattr(row, name)
            assert f[name] == pytest.approx(float(expected), rel=1e-6,
                                            abs=1e-6), (
                f"parity break at {t.txn_id} feature={name}: "
                f"online={f[name]} offline={expected}")


# ------------------------------------------------------------- scenarios
def test_parity_random_stream_offline_vs_inmemory():
    """300-event random stream: every feature, every row, both engines.

    Events use the platform's second-resolution contract (whole-second
    timestamps, like the producer and the synthetic dataset).
    """
    import random
    rnd = random.Random(7)
    txns, ts = [], T0
    cards = [f"C-{i}" for i in range(12)]
    users = [f"U-{i}" for i in range(8)]
    for i in range(300):
        ts += timedelta(seconds=max(1, int(rnd.expovariate(1 / 420.0))))
        txns.append(txn(i, ts, card=rnd.choice(cards),
                        user=rnd.choice(users),
                        amount=round(rnd.lognormvariate(3.4, 1.0), 2),
                        device=rnd.choice(["D-1", "D-2", "D-3"]),
                        merchant=rnd.choice(["M-1", "M-2", "M-3"])))
    assert_parity(offline_features(txns), InMemoryFeatureStore(), txns)


def test_parity_exact_window_boundaries():
    """Events at EXACTLY 600s / 3600s / 86400s must count identically
    offline and online."""
    specs = [(600, 10), (3600, 50), (86400, 100)]
    txns = [txn(0, T0, card="C-b", user="U-b", amount=10)]
    for i, (gap, amt) in enumerate(specs, start=1):
        txns.append(txn(i, T0 + timedelta(seconds=gap * i), card="C-b",
                        user="U-b", amount=amt))
    assert_parity(offline_features(txns), InMemoryFeatureStore(), txns)


def test_parity_same_timestamp_events():
    """Two events on one card at the SAME second: offline rolling window
    (closed='left') does not see the peer; online must match."""
    txns = [txn(0, T0, card="C-s", user="U-s", amount=100),
            txn(1, T0, card="C-s", user="U-s", amount=200),
            txn(2, T0 + timedelta(seconds=30), card="C-s", user="U-s",
                amount=50)]
    assert_parity(offline_features(txns), InMemoryFeatureStore(), txns)


def test_parity_card_silent_over_a_week():
    """DEFECT-01 regression: a card returning after >7 days must report
    the TRUE time_since_prev offline AND online (no 7-day capping)."""
    gap = timedelta(days=9, hours=3)
    txns = [txn(0, T0, card="C-old", user="U-old", amount=100),
            txn(1, T0 + gap, card="C-old", user="U-old", amount=75,
                device="D-2")]
    assert_parity(offline_features(txns), InMemoryFeatureStore(), txns)
    f = InMemoryFeatureStore().compute_and_update(txns[1]) if False else None
    # explicit business meaning: ~9 days, not the 24h default
    store = InMemoryFeatureStore()
    store.compute_and_update(txns[0])
    f2 = store.compute_and_update(txns[1])
    assert f2["card_time_since_prev_sec"] == pytest.approx(
        gap.total_seconds())


def test_parity_inmemory_vs_redis_fakeredis():
    """The two ONLINE backends must agree with each other, event by event."""
    import random
    rnd = random.Random(11)
    txns, ts = [], T0
    for i in range(120):
        ts += timedelta(seconds=max(1, int(rnd.expovariate(1 / 300.0))))
        txns.append(txn(i, ts, card=rnd.choice(["C-A", "C-B"]),
                        user="U-A",
                        amount=round(rnd.lognormvariate(3.0, 0.9), 2),
                        device=rnd.choice(["D-1", "D-2"]),
                        merchant=rnd.choice(["M-1", "M-2"])))

    mem = InMemoryFeatureStore()
    red = _redis_store()

    f_mem = [mem.compute_and_update(t) for t in txns]
    f_red = [red.compute_and_update(t) for t in txns]
    for a, b in zip(f_mem, f_red):
        for name in FEATURES:
            assert a[name] == pytest.approx(b[name], rel=1e-6, abs=1e-6), (
                f"backend mismatch on {name}: memory={a[name]} "
                f"redis={b[name]}")


def test_parity_redis_vs_offline_full():
    """Redis backend must also match the offline builder exactly."""
    import random
    rnd = random.Random(13)
    txns, ts = [], T0
    for i in range(150):
        ts += timedelta(seconds=max(1, int(rnd.expovariate(1 / 500.0))))
        txns.append(txn(i, ts, card=rnd.choice(["C-K", "C-L", "C-M"]),
                        user="U-K",
                        amount=round(rnd.lognormvariate(3.2, 1.0), 2),
                        device=rnd.choice(["D-1", "D-9"]),
                        merchant=rnd.choice(["M-1", "M-4"])))
    assert_parity(offline_features(txns), _redis_store(), txns)


# ------------------------------------------------------------- helpers
def _redis_store():
    """RedisFeatureStore backed by fakeredis (no server needed)."""
    import fakeredis
    import redis as redis_lib

    class _FakeFactory:
        @classmethod
        def from_url(cls, url, **kw):
            return fakeredis.FakeStrictRedis(decode_responses=True)

    original = redis_lib.Redis.from_url
    redis_lib.Redis.from_url = _FakeFactory.from_url
    try:
        store = RedisFeatureStore("redis://fakeredis:6379/0")
    finally:
        redis_lib.Redis.from_url = original
    return store
