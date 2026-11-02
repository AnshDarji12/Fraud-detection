"""MODEL QUALITY GATES (BA-06, BA-09) - the REAL trained model on the
REAL time-based hold-out split. These are release gates: if any fails,
the model must not ship.

Floors carry margin against the evidence numbers in
artifacts/qa/probe_evidence.json so the gate tests the contract,
not dataset luck.
"""
import json

import numpy as np
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)

from src.common.features import FEATURES

MIN_ROC_AUC = 0.90          # evidence: 0.986
MIN_PR_AUC = 0.50           # evidence: 0.934
MIN_RECALL_AT_REVIEW = 0.80  # evidence: 0.923
MIN_PRECISION_AT_REVIEW = 0.70  # evidence: 0.878
MIN_LEGIT_APPROVAL = 0.97   # BA-06 false-positive budget (evidence: 0.9966)
MAX_BLOCK_RATE = 0.05       # evidence: 0.026
MIN_FRAUD_NON_APPROVE = 0.85  # BA-09 (evidence: 0.923)


def _predictions(test_split, bundle):
    p = bundle.model.predict_proba(test_split[FEATURES])[:, 1]
    y = test_split["is_fraud"].astype(int).values
    return y, p


def _decisions(p, thr_review, thr_block):
    return np.where(p >= thr_block, "BLOCK",
                    np.where(p >= thr_review, "REVIEW", "APPROVE"))


# ---------------------------------------------------------------- gates
def test_roc_auc_gate(test_split, bundle):
    y, p = _predictions(test_split, bundle)
    auc = roc_auc_score(y, p)
    assert auc >= MIN_ROC_AUC, f"ROC-AUC {auc:.4f} < gate {MIN_ROC_AUC}"


def test_pr_auc_gate(test_split, bundle):
    y, p = _predictions(test_split, bundle)
    ap = average_precision_score(y, p)
    assert ap >= MIN_PR_AUC, f"PR-AUC {ap:.4f} < gate {MIN_PR_AUC}"


def test_recall_at_review_gate(test_split, bundle, metrics):
    y, p = _predictions(test_split, bundle)
    pred = (p >= metrics["review_threshold"]).astype(int)
    rec = recall_score(y, pred, zero_division=0)
    assert rec >= MIN_RECALL_AT_REVIEW, (
        f"recall@review {rec:.4f} < gate {MIN_RECALL_AT_REVIEW}")


def test_precision_at_review_gate(test_split, bundle, metrics):
    y, p = _predictions(test_split, bundle)
    pred = (p >= metrics["review_threshold"]).astype(int)
    prec = precision_score(y, pred, zero_division=0)
    assert prec >= MIN_PRECISION_AT_REVIEW, (
        f"precision@review {prec:.4f} < gate {MIN_PRECISION_AT_REVIEW}")


def test_ba06_false_positive_budget(test_split, bundle, metrics):
    """Legitimate customers must not be molested: >= 97% of legit txns
    auto-approved on the hold-out split."""
    y, p = _predictions(test_split, bundle)
    dec = _decisions(p, metrics["review_threshold"],
                     metrics["block_threshold"])
    legit = y == 0
    approval = (dec[legit] == "APPROVE").mean()
    assert approval >= MIN_LEGIT_APPROVAL, (
        f"legit approval {approval:.4%} < budget {MIN_LEGIT_APPROVAL:.0%}")


def test_ba09_fraud_never_silently_approved(test_split, bundle, metrics):
    y, p = _predictions(test_split, bundle)
    dec = _decisions(p, metrics["review_threshold"],
                     metrics["block_threshold"])
    caught = (dec[y == 1] != "APPROVE").mean()
    assert caught >= MIN_FRAUD_NON_APPROVE, (
        f"fraud non-approve {caught:.2%} < gate {MIN_FRAUD_NON_APPROVE:.0%}")


def test_block_rate_within_business_tolerance(test_split, bundle, metrics):
    """Blocking too many txns destroys checkout conversion."""
    _, p = _predictions(test_split, bundle)
    br = (p >= metrics["block_threshold"]).mean()
    assert br <= MAX_BLOCK_RATE, f"block rate {br:.2%} > {MAX_BLOCK_RATE:.0%}"


# ------------------------------------------------------------- hygiene
def test_scores_are_finite_and_bounded(test_split, bundle):
    _, p = _predictions(test_split, bundle)
    assert np.isfinite(p).all(), "model produced NaN/Inf scores"
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_thresholds_are_ordered(metrics):
    assert 0.0 < metrics["review_threshold"] < metrics["block_threshold"] < 1.0


def test_model_is_deterministic(test_split, bundle):
    """Same input twice -> identical scores (no hidden state)."""
    X = test_split[FEATURES].head(200)
    p1 = bundle.model.predict_proba(X)[:, 1]
    p2 = bundle.model.predict_proba(X)[:, 1]
    assert np.array_equal(p1, p2)


def test_metrics_json_matches_reality(test_split, bundle):
    """metrics.json (served to ops dashboards) must not drift from the
    artifact it claims to describe."""
    with open("artifacts/models/metrics.json") as f:
        m = json.load(f)
    y, p = _predictions(test_split, bundle)
    auc = roc_auc_score(y, p)
    assert abs(auc - m["test_roc_auc"]) < 0.01, (
        f"metrics.json claims {m['test_roc_auc']:.4f}, actual {auc:.4f}")
