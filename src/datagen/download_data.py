"""Download real public fraud datasets via Kaggle (OPTIONAL data source).

The platform runs out-of-the-box with the synthetic generator; use this
script to add REAL labeled data for training realism.

Datasets (all free, Kaggle account + API token required):
  - creditcard  : ULB Credit Card Fraud (284,807 txns, 0.172% fraud)
  - ieee-cis    : IEEE-CIS Fraud Detection (~590K txns, ~3.5% fraud)

Setup:
  1. kaggle.com -> Account -> Create New API Token -> ~/.kaggle/kaggle.json
  2. pip install kaggle
  3. python -m src.datagen.download_data --dataset creditcard
"""
import argparse
import hashlib
import os
import subprocess

import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"


def kaggle_download(spec: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", spec,
                    "-p", RAW_DIR, "--unzip"], check=True)
    return RAW_DIR


def map_creditcard() -> pd.DataFrame:
    """Map ULB creditcard.csv (V1..V28 PCA features) to our schema.

    The PCA features are dropped; we keep the behavioural fields so the
    SAME feature engineering applies. user/card entities are derived
    deterministically so per-entity velocity features still work.
    """
    path = os.path.join(RAW_DIR, "creditcard.csv")
    df = pd.read_csv(path)
    t0 = pd.Timestamp("2026-09-01 00:00:00")
    out = pd.DataFrame({
        "event_ts": t0 + pd.to_timedelta(df["Time"], unit="s"),
        "amount": df["Amount"].clip(lower=1.0),
        "is_fraud": df["Class"],
    })
    # deterministic pseudo-entities from the first PCA component
    out["card_id"] = df["V1"].rank(method="first").astype(int).map(
        lambda i: f"C-RC-{i % 5000:05d}")
    out["user_id"] = out["card_id"].map(lambda c: "U-RC-" + c[-5:])
    out["merchant_id"] = df["V2"].rank(method="first").astype(int).map(
        lambda i: f"M-RC-{i % 300:04d}")
    out["merchant_cat"] = "replay"
    out["device_id"] = "D-REPLAY"
    out["channel"] = "replay"
    out["ip"] = ""
    out["lat"] = None
    out["lon"] = None
    out["currency"] = "USD"
    return out.sort_values("event_ts").reset_index(drop=True)


DATASETS = {
    "creditcard": ("mlg-ulb/creditcardfraud", map_creditcard),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    args = ap.parse_args()

    spec, mapper = DATASETS[args.dataset]
    print(f"[..] downloading {spec} ...")
    kaggle_download(spec)
    df = mapper()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.dataset}_transactions.csv")
    df.to_csv(out_path, index=False)
    print(f"[ok] wrote {len(df)} rows -> {out_path} "
          f"({df['is_fraud'].mean() * 100:.3f}% fraud)")


if __name__ == "__main__":
    main()
