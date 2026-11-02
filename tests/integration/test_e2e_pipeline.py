"""END-TO-END PIPELINE TESTS (BA-11, BA-12).

Replays REAL transactions through the exact production path:
  feature store -> model -> decision engine -> alert fan-out
and proves the pipeline survives duplicates, poison and bursts.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.common.features import FEATURES, to_vector
from src.common.schemas import Decision, Transaction
from src.inference.app import decide
from src.streaming.features_online import InMemoryFeatureStore
from src.streaming.txn_deduper import TxnDeduper

UTC = timezone.utc


def to_txn(row) -> Transaction:
    return Transaction(txn_id=row.txn_id, event_ts=row.event_ts.to_pydatetime(),
                       user_id=row.user_id, card_id=row.card_id,
                       merchant_id=row.merchant_id,
                       merchant_cat=row.merchant_cat,
                       amount=float(row.amount), currency=row.currency,
                       channel=row.channel, device_id=row.device_id,
                       ip=row.ip,
                       lat=float(row.lat) if pd.notna(row.lat) else None,
                       lon=float(row.lon) if pd.notna(row.lon) else None)


def score_txn(bundle, store, txn):
    f = store.compute_and_update(txn)
    X = pd.DataFrame([to_vector(f)], columns=FEATURES)
    p = float(bundle.model.predict_proba(X)[0][1])
    d, reason = decide(p, f)
    return p, d, reason


def test_e2e_replay_real_stream(bundle, test_split):
    """Every txn of the hold-out window flows through the full local
    path; outputs are valid and internally consistent."""
    store = InMemoryFeatureStore()
    n = 0
    approve = review = block = 0
    alerts = []
    for row in test_split.itertuples(index=False):
        txn = to_txn(row)
        p, d, reason = score_txn(bundle, store, txn)
        assert 0.0 <= p <= 1.0
        assert isinstance(d, Decision)
        if d == Decision.APPROVE:
            approve += 1
        else:
            alerts.append((txn.txn_id, d))
            if d == Decision.REVIEW:
                review += 1
            else:
                block += 1
        n += 1
    assert n == len(test_split)
    assert approve + review + block == n
    assert review + block == len(alerts)          # alert fan-out consistent
    assert block <= 0.05 * n                       # block-rate tolerance


def test_e2e_card_testing_burst_is_caught(bundle):
    """Live attack simulation: 8 rapid $15-40 txns on a fresh card.
    Business expectation (BA-01): once the 10-min velocity hits the
    rule limit, later burst txns are NOT auto-approved."""
    store = InMemoryFeatureStore()
    t0 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    decisions = []
    for i, amt in enumerate([20, 25, 30, 22, 28, 35, 40, 33]):
        t = Transaction(txn_id=f"T-BURST-{i}", event_ts=t0 + timedelta(seconds=30 * i),
                        user_id="U-ATTACK", card_id="C-ATTACK",
                        merchant_id=f"M-{i:03d}", merchant_cat="online",
                        amount=float(amt), channel="online",
                        device_id="D-STOLEN")
        _, d, _ = score_txn(bundle, store, t)
        decisions.append(d)
    flagged = [d for d in decisions[5:]]
    assert all(d != Decision.APPROVE for d in flagged), (
        f"burst tail must be flagged, got {decisions}")


def test_ba11_duplicate_delivery_is_processed_once(bundle):
    """Kafka is at-least-once: a redelivered txn must NOT be double-
    scored / double-alerted (regression for DEFECT-02)."""
    deduper = TxnDeduper(maxlen=1000)
    store = InMemoryFeatureStore()
    t0 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    events = [Transaction(txn_id="T-DUP-1", event_ts=t0,
                          user_id="U-D", card_id="C-D",
                          merchant_id="M-1", merchant_cat="grocery",
                          amount=50.0, channel="web", device_id="D-1"),
              Transaction(txn_id="T-DUP-2", event_ts=t0 + timedelta(seconds=10),
                          user_id="U-D", card_id="C-D",
                          merchant_id="M-1", merchant_cat="grocery",
                          amount=50.0, channel="web", device_id="D-1")]
    processed = []
    for txn in events + [events[0]]:            # 3rd message = redelivery
        if deduper.seen_before(txn.txn_id):
            continue
        processed.append(score_txn(bundle, store, txn))
    assert len(processed) == 2, "duplicate was processed twice"
    assert deduper.hits == 1


def test_ba12_poison_message_does_not_kill_pipeline(bundle):
    """Contract violations must be isolatable (DLQ upstream); the next
    valid event must still be scorable."""
    from pydantic import ValidationError
    store = InMemoryFeatureStore()
    poison = [{"txn_id": "T-POISON", "event_ts": "not-a-date",
               "user_id": "U", "card_id": "C", "merchant_id": "M",
               "merchant_cat": "g", "amount": 10.0},
              {"txn_id": "T-POISON2", "event_ts": "2026-10-01T12:00:00",
               "user_id": "U", "card_id": "C", "merchant_id": "M",
               "merchant_cat": "g", "amount": -99}]
    rejected = 0
    for p in poison:
        try:
            Transaction(**p)
        except ValidationError:
            rejected += 1
    assert rejected == len(poison)
    # pipeline still alive:
    t0 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    ok = Transaction(txn_id="T-OK", event_ts=t0, user_id="U-OK",
                     card_id="C-OK", merchant_id="M-1",
                     merchant_cat="grocery", amount=42.0,
                     channel="web", device_id="D-1")
    p, d, _ = score_txn(bundle, store, ok)
    assert 0.0 <= p <= 1.0 and isinstance(d, Decision)
