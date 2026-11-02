"""DATA QUALITY TESTS - the dataset is a production artifact too.

A fraud model is only as good as its data: these checks guard the
contract of data/processed/demo_transactions.csv (and by extension the
Kaggle replay path feeding it).
"""
import pandas as pd

VALID_CATEGORIES = {"grocery", "fuel", "restaurant", "electronics",
                    "travel", "clothing", "pharmacy", "utilities",
                    "entertainment", "online"}
VALID_CHANNELS = {"mobile_app", "web", "pos", "contactless", "online"}
REQUIRED_COLS = ["txn_id", "event_ts", "user_id", "card_id", "merchant_id",
                 "merchant_cat", "amount", "currency", "channel",
                 "device_id", "ip", "lat", "lon", "is_fraud"]


def test_required_columns_present(df):
    missing = set(REQUIRED_COLS) - set(df.columns)
    assert not missing, f"contract violation, missing: {missing}"


def test_no_nulls_in_required_fields(df):
    n = df[REQUIRED_COLS].isna().sum()
    # lat/lon may be optional in the contract; everything else is NOT
    hard = n.drop(index=["lat", "lon"])
    assert (hard == 0).all(), f"unexpected nulls: {hard[hard > 0].to_dict()}"


def test_no_duplicate_txn_ids(df):
    dups = df["txn_id"].duplicated().sum()
    assert dups == 0, f"{dups} duplicate txn_id values"


def test_amounts_positive_and_bounded(df):
    assert (df["amount"] > 0).all(), "amount <= 0 found"
    assert df["amount"].max() < 100_000, "implausible amount found"


def test_timestamps_sorted_and_in_span(df):
    ts = df["event_ts"]
    assert ts.is_monotonic_increasing, "stream must be time-sorted"
    span_days = (ts.max() - ts.min()).days
    assert 10 <= span_days <= 31, f"span {span_days}d outside design band"


def test_fraud_rate_in_design_band(df):
    r = df["is_fraud"].mean()
    assert 0.01 <= r <= 0.06, f"fraud rate {r:.2%} outside 1-6% band"


def test_labels_binary(df):
    assert set(df["is_fraud"].unique()) <= {0, 1}


def test_categoricals_within_whitelist(df):
    bad_cat = set(df["merchant_cat"]) - VALID_CATEGORIES
    bad_chan = set(df["channel"]) - VALID_CHANNELS
    assert not bad_cat, f"unknown merchant_cat: {bad_cat}"
    assert not bad_chan, f"unknown channel: {bad_chan}"


def test_pii_fields_masked_or_internal(df):
    """Synthetic IPs stay in RFC1918 space; no real customer PII ships."""
    ips = df["ip"].astype(str)
    assert ips.str.startswith("10.").all(), "non-internal IP found"


def test_cards_map_to_single_user(df):
    """A card belongs to exactly one user (join-integrity invariant)."""
    n_users_per_card = df.groupby("card_id")["user_id"].nunique()
    assert (n_users_per_card <= 1).all(), "card shared across users"
