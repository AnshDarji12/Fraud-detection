"""Synthetic transaction generator with REALISTIC fraud patterns.

Produces the live dataset for the platform:
  - normal traffic: per-user daily rhythm, lognormal amounts,
    stable device + home geo
  - fraud pattern A: card-testing velocity burst (6-12 txns in minutes)
  - fraud pattern B: new device + night time + high amount
  - fraud pattern C: rare merchant + high amount (account takeover vibe)

Output CSV matches src/common/schemas.py + an `is_fraud` label column
(labels are NOT streamed with transactions - the producer strips them
and publishes them late on txn.labels, like real chargeback data).
"""
import argparse
import hashlib
import math
import os
import random
from datetime import datetime, timedelta

import pandas as pd

CATEGORIES = ["grocery", "fuel", "restaurant", "electronics", "travel",
              "clothing", "pharmacy", "utilities", "entertainment", "online"]
CHANNELS = ["mobile_app", "web", "pos", "contactless"]
RARE_MERCHANTS = ["M-RARE-01", "M-RARE-02", "M-RARE-03", "M-RARE-04"]


def _home_geo(seed: int):
    rnd = random.Random(seed)
    return rnd.uniform(8.0, 30.0), rnd.uniform(68.0, 88.0)  # India-ish box


def generate(rows: int, start: datetime, days: int, seed: int = 42) -> pd.DataFrame:
    rnd = random.Random(seed)
    n_users = max(50, rows // 50)

    users = []
    for i in range(n_users):
        lat, lon = _home_geo(seed + i)
        users.append({
            "user_id": f"U-{i:05d}",
            "cards": [f"C-{i:05d}-{c}" for c in range(rnd.randint(1, 2))],
            "device": f"D-{i:05d}-{rnd.randint(0, 9)}",
            "home_lat": lat,
            "home_lon": lon,
        })

    end = start + timedelta(days=days)
    records = []

    def add_txn(u, ts, amount, merchant_id, cat, device, is_fraud, channel=None):
        ts = ts.replace(microsecond=0)  # keep ISO format uniform
        records.append({
            "txn_id": f"T-{len(records):08d}",
            "event_ts": ts.isoformat(),
            "user_id": u["user_id"],
            "card_id": rnd.choice(u["cards"]) if not is_fraud else u["cards"][0],
            "merchant_id": merchant_id,
            "merchant_cat": cat,
            "amount": round(min(max(amount, 1.0), 8000.0), 2),
            "currency": "USD",
            "channel": channel or rnd.choice(CHANNELS),
            "device_id": device,
            "ip": f"10.{rnd.randint(0, 255)}.{rnd.randint(0, 255)}.{rnd.randint(1, 254)}",
            "lat": round(u["home_lat"] + rnd.gauss(0, 0.05), 5),
            "lon": round(u["home_lon"] + rnd.gauss(0, 0.05), 5),
            "is_fraud": int(is_fraud),
        })

    # ---------- normal traffic ----------
    while len(records) < int(rows * 0.97):
        u = rnd.choice(users)
        day_offset = rnd.uniform(0, days - 0.001)
        ts = start + timedelta(days=day_offset)
        # daytime-weighted hours
        hour = min(23, max(0, int(rnd.gauss(14, 5))))
        ts = ts.replace(hour=hour, minute=rnd.randint(0, 59), second=rnd.randint(0, 59),
                        microsecond=0)
        if ts >= end:
            continue
        amount = math.exp(rnd.gauss(3.4, 1.0))  # median ~30
        merchant_idx = int(hashlib.md5(u["user_id"].encode()).hexdigest()[:6], 16) % len(CATEGORIES)
        add_txn(u, ts, amount,
                f"M-{merchant_idx % 60:03d}", CATEGORIES[merchant_idx % len(CATEGORIES)],
                u["device"], is_fraud=False)

    # ---------- fraud injections (~3%) ----------
    n_fraud = rows - len(records)
    patterns = ["velocity", "device_night", "rare_merchant"]

    for k in range(n_fraud):
        u = rnd.choice(users)
        base = start + timedelta(seconds=rnd.uniform(0, days * 86400 * 0.95))
        pattern = patterns[k % len(patterns)]

        if pattern == "velocity":
            burst_len = rnd.randint(6, 10)
            for j in range(burst_len):
                if len(records) >= rows:
                    break
                ts = base + timedelta(seconds=rnd.uniform(20, 480) * j)
                add_txn(u, ts, rnd.uniform(15, 250),
                        f"M-{rnd.randint(0, 59):03d}", rnd.choice(CATEGORIES),
                        u["device"], is_fraud=True, channel="online")
        elif pattern == "device_night":
            for j in range(rnd.randint(1, 3)):
                if len(records) >= rows:
                    break
                ts = base + timedelta(minutes=rnd.randint(0, 40))
                ts = ts.replace(hour=rnd.randint(1, 5), minute=rnd.randint(0, 59))
                add_txn(u, ts, rnd.uniform(400, 3000),
                        f"M-{rnd.randint(0, 59):03d}", "electronics",
                        f"D-UNKNOWN-{rnd.randint(100, 999)}", is_fraud=True,
                        channel="web")
        else:  # rare merchant
            for j in range(rnd.randint(1, 2)):
                if len(records) >= rows:
                    break
                ts = base + timedelta(minutes=rnd.randint(0, 30))
                add_txn(u, ts, rnd.uniform(600, 3500),
                        rnd.choice(RARE_MERCHANTS), "online", u["device"],
                        is_fraud=True, channel="online")

    df = pd.DataFrame(records)
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    df = df.sort_values("event_ts").reset_index(drop=True)
    # re-key txn ids in time order (stable, readable stream)
    df["txn_id"] = [f"T-{i:08d}" for i in range(len(df))]
    return df


def main():
    ap = argparse.ArgumentParser(description="Synthetic transaction generator")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--start", type=str, default="2026-10-01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/processed/demo_transactions.csv")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    df = generate(args.rows, start, args.days, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    fraud_pct = 100 * df["is_fraud"].mean()
    print(f"[ok] wrote {len(df)} rows -> {args.out} "
          f"({df['user_id'].nunique()} users, {fraud_pct:.2f}% fraud)")


if __name__ == "__main__":
    main()
