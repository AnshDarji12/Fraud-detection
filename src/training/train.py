"""Train the fraud model and manage its lifecycle in MLflow.

Pipeline:
  1. load labeled transactions (CSV from generator / Kaggle replay)
  2. build offline features (identical contract to online store)
  3. TIME-BASED split 70/15/15 (no random split - prevents leakage)
  4. XGBoost with class imbalance handling (scale_pos_weight)
  5. choose REVIEW/BLOCK thresholds on validation (precision-targeted)
  6. log everything to MLflow + register as `fraud_xgb` -> Production
  7. also save a local joblib copy so the API can start without MLflow
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

from src.common.features import FEATURES
from src.training.features import build_features

SEED = 42


def time_split(df: pd.DataFrame, ratios=(0.7, 0.15, 0.15)):
    df = df.sort_values("event_ts").reset_index(drop=True)
    n = len(df)
    i1, i2 = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def pick_thresholds(y_val, p_val, target_precision=0.90):
    """REVIEW = max-F1 threshold; BLOCK = smallest threshold that keeps
    precision >= target (fall back to 0.95 if unreachable)."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    prec, rec, thr = prec[:-1], rec[:-1], thr
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    review_thr = float(thr[int(np.argmax(f1))])

    mask = prec >= target_precision
    block_thr = float(thr[mask].min()) if mask.any() else 0.95
    if block_thr <= review_thr:
        block_thr = min(0.98, review_thr + 0.2)
    return review_thr, block_thr


def report(name, y, p, thr_review, thr_block):
    pred = (p >= thr_review).astype(int)
    blocked = (p >= thr_block).astype(int)
    return {
        f"{name}_pr_auc": float(average_precision_score(y, p)),
        f"{name}_roc_auc": float(roc_auc_score(y, p)),
        f"{name}_recall@review": float(recall_score(y, pred, zero_division=0)),
        f"{name}_precision@review": float(precision_score(y, pred, zero_division=0)),
        f"{name}_block_rate": float(blocked.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/demo_transactions.csv")
    ap.add_argument("--no-mlflow", action="store_true",
                    help="skip MLflow tracking/registry (local run)")
    args = ap.parse_args()

    print(f"[..] loading {args.csv}")
    df = pd.read_csv(args.csv, parse_dates=["event_ts"])
    feats = build_features(df)
    print(f"[ok] features built: {FEATURES}")

    train, val, test = time_split(feats)
    X_tr, y_tr = train[FEATURES], train["is_fraud"].astype(int)
    X_va, y_va = val[FEATURES], val["is_fraud"].astype(int)
    X_te, y_te = test[FEATURES], test["is_fraud"].astype(int)
    print(f"[ok] split train={len(X_tr)} val={len(X_va)} test={len(X_te)} "
          f"(fraud {y_tr.mean():.2%}/{y_va.mean():.2%}/{y_te.mean():.2%})")

    spw = max(1.0, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw, eval_metric="aucpr",
        random_state=SEED, n_jobs=-1, tree_method="hist",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    p_val = model.predict_proba(X_va)[:, 1]
    p_test = model.predict_proba(X_te)[:, 1]
    thr_review, thr_block = pick_thresholds(y_va, p_val)

    metrics = {"review_threshold": thr_review, "block_threshold": thr_block}
    metrics.update(report("val", y_va, p_val, thr_review, thr_block))
    metrics.update(report("test", y_te, p_test, thr_review, thr_block))

    print("\n===== MODEL REPORT =====")
    for k, v in metrics.items():
        print(f"  {k:26s} {v:.4f}" if isinstance(v, float) else f"  {k:26s} {v}")

    os.makedirs("artifacts/models", exist_ok=True)
    model_path = "artifacts/models/fraud_xgb.joblib"
    joblib.dump(model, model_path)
    with open("artifacts/models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[ok] saved local model -> {model_path}")

    if not args.no_mlflow:
        _mlflow_log(model, metrics, X_tr.head(200))


def _mlflow_log(model, metrics, sample_X):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow.models import infer_signature
        from src.common.config import settings
    except ImportError:
        print("[warn] mlflow not installed - skipping registry")
        return

    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run(run_name="xgb_fraud") as run:
        mlflow.log_param("model", "XGBClassifier")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("features", FEATURES)
        mlflow.log_metrics({k: v for k, v in metrics.items()})
        mlflow.sklearn.log_model(
            model, name="model",
            signature=infer_signature(sample_X, model.predict_proba(sample_X)[:, 1]),
        )
        run_id = run.info.run_id
    print(f"[ok] MLflow run {run_id}")

    try:
        client = MlflowClient(tracking_uri=settings.mlflow_uri)
        try:
            mv = client.create_model_version(
                settings.model_name, f"runs:/{run_id}/model")
        except Exception:
            mlflow.register_model(f"runs:/{run_id}/model", settings.model_name)
            latest = client.get_latest_versions(settings.model_name, ["None"])
            mv = latest[0]
        client.transition_model_version_stage(
            settings.model_name, mv.version, settings.model_stage,
            archive_existing_versions=True)
        print(f"[ok] model v{mv.version} -> stage {settings.model_stage}")
    except Exception as e:
        print(f"[warn] registry transition failed ({e}) - "
              f"promote manually in MLflow UI at {settings.mlflow_uri}")


if __name__ == "__main__":
    main()
