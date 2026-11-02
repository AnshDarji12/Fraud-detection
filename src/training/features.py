"""OFFLINE feature engineering - must match src/common/features.py semantics.

Every windowed feature uses `closed='left'` rolling windows so the value
at row t is computed from PRIOR transactions only - exactly what the
online feature store produces at scoring time.
"""
import pandas as pd

from src.common.features import DEFAULT_TIME_SINCE_PREV_SEC, base_features

HOUR = 3600
DAY = 24 * HOUR


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Input: DataFrame with columns
    [event_ts, user_id, card_id, merchant_id, device_id, amount, is_fraud].
    Output: same rows + all FEATURES columns (and keeps is_fraud).
    """
    out = df.copy().sort_values("event_ts").reset_index(drop=True)
    out["event_ts"] = pd.to_datetime(out["event_ts"])

    # --- transaction-only features -------------------------------------
    base = pd.DataFrame({
        "amount": out["amount"].astype(float),
        "hour_of_day": out["event_ts"].dt.hour,
    })
    b = [base_features(r.amount, r.hour_of_day) for r in base.itertuples(index=False)]
    for name in ("amount", "log_amount", "hour_of_day", "is_night"):
        out[name] = [row[name] for row in b]

    out = _card_features(out)
    out = _user_features(out)
    return out


def _card_features(out: pd.DataFrame) -> pd.DataFrame:
    ts = out.set_index("event_ts")

    cnt10 = pd.Series(index=out.index, dtype=float)
    cnt1h = pd.Series(index=out.index, dtype=float)
    sum1h = pd.Series(index=out.index, dtype=float)
    tprev = pd.Series(index=out.index, dtype=float)
    devchg = pd.Series(index=out.index, dtype=float)
    merchchg = pd.Series(index=out.index, dtype=float)

    for _, gd in out.groupby("card_id", sort=False):
        g = gd.set_index("event_ts")
        amt = g["amount"].astype(float)

        cnt10.loc[gd.index] = amt.rolling("600s", closed="left").count().values
        cnt1h.loc[gd.index] = amt.rolling("3600s", closed="left").count().values
        sum1h.loc[gd.index] = amt.rolling("3600s", closed="left").sum().values

        delta = g.index.to_series().diff().dt.total_seconds()
        tprev.loc[gd.index] = delta.values

        prev_dev = g["device_id"].shift(1)
        devchg.loc[gd.index] = (
            (g["device_id"] != prev_dev)
            & (delta <= DAY)
        ).astype(float).values

        prev_merch = g["merchant_id"].shift(1)
        merchchg.loc[gd.index] = (
            (g["merchant_id"] != prev_merch)
            & (delta <= HOUR)
        ).astype(float).values

    out["card_txn_count_10m"] = cnt10.fillna(0)
    out["card_txn_count_1h"] = cnt1h.fillna(0)
    out["card_amount_sum_1h"] = sum1h.fillna(0)
    out["card_time_since_prev_sec"] = tprev.fillna(DEFAULT_TIME_SINCE_PREV_SEC)
    out["card_device_changed"] = devchg.fillna(0)
    out["card_merchant_changed"] = merchchg.fillna(0)
    return out


def _user_features(out: pd.DataFrame) -> pd.DataFrame:
    cnt24 = pd.Series(index=out.index, dtype=float)
    avg7d = pd.Series(index=out.index, dtype=float)

    for _, gd in out.groupby("user_id", sort=False):
        g = gd.set_index("event_ts")
        amt = g["amount"].astype(float)

        cnt24.loc[gd.index] = amt.rolling("86400s", closed="left").count().values
        avg7d.loc[gd.index] = amt.rolling("604800s", closed="left").mean().values

    out["user_txn_count_24h"] = cnt24.fillna(0)
    out["user_amount_avg_7d"] = avg7d.fillna(0)
    return out
