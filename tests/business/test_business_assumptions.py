"""BUSINESS ASSUMPTION TESTS (BA-01 .. BA-05, BA-08, BA-12).

Each test validates one business assumption of the fraud platform
against the REAL 20k-transaction dataset - the numbers are evidence,
not opinions. Thresholds carry margin so they test the assumption,
not the randomness of one dataset.

Reference evidence (probe run): see artifacts/qa/probe_evidence.json
"""
import pydantic
import pytest

from src.common.schemas import Transaction


# ---------------------------------------------------------------- BA-01
def test_ba01_velocity_bursts_are_fraud_concentrated(feats):
    """Card-testing bursts: >=5 txns / 10 min on one card is a fraud signal."""
    bursty = feats["card_txn_count_10m"] >= 5
    assert bursty.sum() >= 5, "expected velocity bursts to exist in data"
    rate_burst = feats.loc[bursty, "is_fraud"].mean()
    rate_all = feats["is_fraud"].mean()
    # evidence: 100% vs 3% overall -> require >= 3x uplift
    assert rate_burst >= 3 * rate_all, (
        f"burst fraud rate {rate_burst:.2%} not >= 3x base {rate_all:.2%}")


# ---------------------------------------------------------------- BA-02
def test_ba02_night_hours_carry_higher_fraud_risk(df):
    """Fraud concentrates at night (22:00-06:00): device-takeover window."""
    hr = df["event_ts"].dt.hour
    night = (hr >= 22) | (hr < 6)
    assert night.sum() > 500, "night slice too small to be meaningful"
    r_night = df.loc[night, "is_fraud"].mean()
    r_day = df.loc[~night, "is_fraud"].mean()
    # evidence: 11.6% vs 1.9% (6.3x) -> require >= 2x
    assert r_night >= 2 * r_day, (
        f"night {r_night:.2%} not >= 2x day {r_day:.2%}")


# ---------------------------------------------------------------- BA-03
def test_ba03_device_change_within_24h_is_takeover_signal(df):
    """A card switching device within 24h is strongly fraud-associated."""
    d = df.sort_values("event_ts")
    prev_dev = d.groupby("card_id")["device_id"].shift(1)
    prev_ts = d.groupby("card_id")["event_ts"].shift(1)
    delta = (d["event_ts"] - prev_ts).dt.total_seconds()
    changed = ((d["device_id"] != prev_dev) & (delta <= 86400)).fillna(False)
    assert changed.sum() > 50, "device-change slice too small"
    r_chg = d.loc[changed, "is_fraud"].mean()
    r_same = d.loc[~changed, "is_fraud"].mean()
    # evidence: 67.3% vs 2.5% (27x) -> require >= 3x
    assert r_chg >= 3 * r_same, (
        f"device-change {r_chg:.2%} not >= 3x device-same {r_same:.2%}")


# ---------------------------------------------------------------- BA-04
def test_ba04_fraud_amounts_skew_high(df):
    """Fraudulent amounts run hotter than legitimate spend."""
    m_fraud = df.loc[df.is_fraud == 1, "amount"].median()
    m_legit = df.loc[df.is_fraud == 0, "amount"].median()
    # evidence: median 181 vs 30 -> require >= 2x
    assert m_fraud >= 2 * m_legit, (
        f"fraud median {m_fraud} not >= 2x legit median {m_legit}")


# ---------------------------------------------------------------- BA-05
def test_ba05_rare_merchants_are_high_risk(df):
    """Transactions at uncommon merchants carry elevated fraud risk."""
    rare = df["merchant_id"].str.startswith("M-RARE")
    assert rare.sum() >= 30, "rare-merchant slice too small"
    r_rare = df.loc[rare, "is_fraud"].mean()
    r_norm = df.loc[~rare, "is_fraud"].mean()
    # evidence: 100% vs 2.6% -> require >= 5x
    assert r_rare >= 5 * r_norm, (
        f"rare-merchant {r_rare:.2%} not >= 5x normal {r_norm:.2%}")


# ---------------------------------------------------------------- BA-08
def test_ba08_labels_never_leak_into_scoring_path(df):
    """The scoring contract must be label-blind: even if a producer bug
    attaches `is_fraud` to a raw event, the Transaction contract drops it
    before scoring (prevents label leakage / target shuffling)."""
    row = df.iloc[0]
    payload = {
        "txn_id": row.txn_id, "event_ts": row.event_ts.isoformat(),
        "user_id": row.user_id, "card_id": row.card_id,
        "merchant_id": row.merchant_id, "merchant_cat": row.merchant_cat,
        "amount": float(row.amount), "currency": "USD", "channel": "web",
        "device_id": row.device_id, "ip": row.ip,
        "is_fraud": 1,  # hostile/buggy extra field
    }
    t = Transaction(**payload)
    dumped = t.model_dump(mode="json")
    assert "is_fraud" not in dumped
    assert "is_fraud" not in str(dumped)


# ---------------------------------------------------------------- BA-12
def test_ba12_poison_payloads_rejected_at_contract():
    """Malformed events must fail AT the contract (-> DLQ upstream),
    never silently enter the feature store."""
    base = dict(txn_id="T-P", event_ts="2026-10-01T12:00:00", user_id="U",
                card_id="C", merchant_id="M", merchant_cat="g")
    bad_payloads = [
        {**base, "amount": -1},                # negative
        {**base, "amount": 0},                 # zero (gt=0 contract)
        {**{k: v for k, v in base.items() if k != "txn_id"}, "amount": 5},
        {**base, "amount": "expensive"},       # wrong type
    ]
    for p in bad_payloads:
        with pytest.raises(pydantic.ValidationError):
            Transaction(**p)
