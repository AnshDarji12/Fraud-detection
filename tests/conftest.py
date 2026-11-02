"""Shared QA fixtures - real dataset, real trained model, real thresholds.

Everything runs WITHOUT Docker/Kafka/Redis: the online feature store's
in-memory backend and the local joblib model make the full path testable
(fakeredis stands in for Redis where the Redis backend itself is tested).
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)  # settings use relative artifact paths

CSV = PROJECT_ROOT / "data" / "processed" / "demo_transactions.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "fraud_xgb.joblib"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "models" / "metrics.json"
CACHE_PATH = PROJECT_ROOT / "artifacts" / "qa" / "offline_feats_cache.joblib"


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    """Raw labeled transactions, time-sorted (the QA ground truth)."""
    d = pd.read_csv(CSV, parse_dates=["event_ts"])
    return d.sort_values("event_ts").reset_index(drop=True)


@pytest.fixture(scope="session")
def feats(df) -> pd.DataFrame:
    """OFFLINE features for the whole dataset (cached between runs)."""
    if CACHE_PATH.exists():
        return joblib.load(CACHE_PATH)
    from src.training.features import build_features
    out = build_features(df)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, CACHE_PATH)
    return out


@pytest.fixture(scope="session")
def test_split(feats) -> pd.DataFrame:
    """Exact hold-out slice used by src/training/train.py (last 15%)."""
    n = len(feats)
    return feats.iloc[int(n * 0.85):].reset_index(drop=True)


@pytest.fixture(scope="session")
def bundle():
    """The REAL trained model (local artifact - MLflow not needed)."""
    from src.inference.model_loader import load_model
    return load_model()


@pytest.fixture(scope="session")
def metrics() -> dict:
    """Thresholds + training-time metrics.json (production contract)."""
    with open(METRICS_PATH) as f:
        return json.load(f)
