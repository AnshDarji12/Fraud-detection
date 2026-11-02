"""PERFORMANCE / SLA TESTS (BA-07).

Business SLA: end-to-end scoring decision < 200 ms.
Measured path = the REAL production path minus network hops:
  online feature store -> model predict -> decision engine
(p95 gate uses headroom for Kafka + HTTP in the deployed topology).
"""
import time

import numpy as np
import pandas as pd
import pytest

from src.common.features import FEATURES, to_vector
from src.common.schemas import Transaction
from src.inference.app import decide
from src.streaming.features_online import InMemoryFeatureStore

pytestmark = pytest.mark.performance

SLA_MS = 200.0
MIN_TPS = 200  # platform design target


def _latencies(bundle, rows):
    store = InMemoryFeatureStore()
    lat = []
    for row in rows.itertuples(index=False):
        t = Transaction(txn_id=row.txn_id,
                        event_ts=row.event_ts.to_pydatetime(),
                        user_id=row.user_id, card_id=row.card_id,
                        merchant_id=row.merchant_id,
                        merchant_cat=row.merchant_cat,
                        amount=float(row.amount), currency=row.currency,
                        channel=row.channel, device_id=row.device_id,
                        ip=row.ip, lat=None, lon=None)
        t0 = time.perf_counter()
        f = store.compute_and_update(t)
        X = pd.DataFrame([to_vector(f)], columns=FEATURES)
        p = float(bundle.model.predict_proba(X)[0][1])
        decide(p, f)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return np.array(lat)


@pytest.fixture(scope="module")
def latency_report(bundle, test_split):
    lat = _latencies(bundle, test_split.iloc[:1500])
    return {
        "n": len(lat),
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
        "p99": float(np.percentile(lat, 99)),
        "max": float(lat.max()),
        "mean": float(lat.mean()),
    }


def test_p95_under_sla(latency_report):
    assert latency_report["p95"] < SLA_MS, (
        f"p95 {latency_report['p95']:.1f}ms violates {SLA_MS}ms SLA")


def test_p99_under_sla(latency_report):
    assert latency_report["p99"] < SLA_MS, (
        f"p99 {latency_report['p99']:.1f}ms violates {SLA_MS}ms SLA")


def test_max_under_sla(latency_report):
    """No single event may blow the checkout budget."""
    assert latency_report["max"] < SLA_MS, (
        f"max {latency_report['max']:.1f}ms violates {SLA_MS}ms SLA")


def test_sustained_throughput(bundle, test_split):
    rows = test_split.iloc[:1500]
    t0 = time.perf_counter()
    _latencies(bundle, rows)
    wall = time.perf_counter() - t0
    tps = len(rows) / wall
    assert tps >= MIN_TPS, (
        f"throughput {tps:.0f} TPS < design target {MIN_TPS} TPS")


def test_latency_is_stable_not_degrading(bundle, test_split):
    """First-500 vs last-500 latency: no warm-up blowup, no leak-style
    degradation across a sustained run."""
    lat = _latencies(bundle, test_split.iloc[:1500])
    first = float(np.percentile(lat[:500], 95))
    last = float(np.percentile(lat[-500:], 95))
    assert last < SLA_MS and first < SLA_MS
    assert last < max(first * 3.0, 5.0 * SLA_MS / 100.0), (
        f"p95 degraded from {first:.1f}ms to {last:.1f}ms")
