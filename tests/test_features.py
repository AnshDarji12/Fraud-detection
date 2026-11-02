from datetime import datetime, timezone

from src.streaming.features_online import InMemoryFeatureStore


def txn(i, card="C-1", user="U-1", ts=None, amount=50.0, device="D-1",
        merchant="M-1"):
    from src.common.schemas import Transaction
    return Transaction(
        txn_id=f"T-{i}", event_ts=ts or datetime(2026, 10, 1, 12, 0,
                                                 tzinfo=timezone.utc),
        user_id=user, card_id=card, merchant_id=merchant,
        merchant_cat="grocery", amount=amount, channel="web", device_id=device,
    )


def dt(h=12, m=0, s=0):
    return datetime(2026, 10, 1, h, m, s, tzinfo=timezone.utc)


def test_first_txn_has_zero_history():
    store = InMemoryFeatureStore()
    f = store.compute_and_update(txn(1))
    assert f["card_txn_count_10m"] == 0
    assert f["card_txn_count_1h"] == 0
    assert f["user_txn_count_24h"] == 0
    assert f["card_time_since_prev_sec"] == 24 * 3600.0


def test_velocity_counts_exclude_current():
    store = InMemoryFeatureStore()
    store.compute_and_update(txn(1, ts=dt(12, 0, 0)))
    f2 = store.compute_and_update(txn(2, ts=dt(12, 5, 0)))
    # one PRIOR txn within 10 min (current excluded)
    assert f2["card_txn_count_10m"] == 1
    assert f2["card_txn_count_1h"] == 1
    f3 = store.compute_and_update(txn(3, ts=dt(12, 8, 0)))
    assert f3["card_txn_count_10m"] == 2


def test_old_txn_falls_out_of_window():
    store = InMemoryFeatureStore()
    store.compute_and_update(txn(1, ts=dt(12, 0, 0)))
    f = store.compute_and_update(txn(2, ts=dt(12, 20, 0)))
    assert f["card_txn_count_10m"] == 0      # 20 min gap > 10 min window
    assert f["card_txn_count_1h"] == 1       # still inside 1h window


def test_amount_sum_1h():
    store = InMemoryFeatureStore()
    store.compute_and_update(txn(1, ts=dt(12, 0, 0), amount=100))
    store.compute_and_update(txn(2, ts=dt(12, 3, 0), amount=250))
    f = store.compute_and_update(txn(3, ts=dt(12, 4, 0), amount=99))
    assert f["card_amount_sum_1h"] == 350.0


def test_device_change_flag():
    store = InMemoryFeatureStore()
    store.compute_and_update(txn(1, ts=dt(12, 0, 0), device="D-1"))
    f = store.compute_and_update(txn(2, ts=dt(12, 2, 0), device="D-9"))
    assert f["card_device_changed"] == 1.0
    # same new device again -> no change
    f = store.compute_and_update(txn(3, ts=dt(12, 3, 0), device="D-9"))
    assert f["card_device_changed"] == 0.0


def test_merchant_change_within_hour_only():
    store = InMemoryFeatureStore()
    store.compute_and_update(txn(1, ts=dt(12, 0, 0), merchant="M-1"))
    f = store.compute_and_update(txn(2, ts=dt(12, 1, 0), merchant="M-2"))
    assert f["card_merchant_changed"] == 1.0
    store2 = InMemoryFeatureStore()
    store2.compute_and_update(txn(1, ts=dt(12, 0, 0), merchant="M-1"))
    f = store2.compute_and_update(txn(2, ts=dt(14, 0, 0), merchant="M-2"))
    assert f["card_merchant_changed"] == 0.0  # gap > 1h


def test_base_features_night_flag():
    store = InMemoryFeatureStore()
    f = store.compute_and_update(txn(1, ts=dt(23, 30)))
    assert f["is_night"] == 1.0
    assert f["hour_of_day"] == 23
