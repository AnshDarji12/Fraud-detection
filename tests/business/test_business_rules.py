"""BUSINESS RULE TESTS - the decision engine's rule layer (BA-09, BA-10).

The model proposes, the rules layer disposes: these tests pin the
business logic that must hold REGARDLESS of what the model says.
"""
from src.common.config import settings
from src.common.schemas import Decision
from src.inference.app import decide


def test_ba09_high_score_is_never_auto_approved():
    """A score at/above the BLOCK threshold can never surface as APPROVE."""
    for p in (settings.block_threshold, 0.99, 1.0):
        d, reason = decide(p, {"card_txn_count_10m": 0})
        assert d == Decision.BLOCK, f"p={p} must BLOCK"
        assert reason  # auditable reason required


def test_ba09_block_takes_precedence_over_velocity_rule():
    """Highest-severity outcome wins: BLOCK beats a REVIEW-level rule."""
    d, reason = decide(settings.block_threshold + 0.01,
                       {"card_txn_count_10m": 99})
    assert d == Decision.BLOCK
    assert "block" in reason.lower()


def test_review_band_between_thresholds():
    lo, hi = settings.review_threshold, settings.block_threshold
    mid = lo + (hi - lo) / 2
    d, _ = decide(mid, {"card_txn_count_10m": 0})
    assert d == Decision.REVIEW


def test_velocity_rule_overrides_model_approve():
    """BA-01 enforced in the engine: burst velocity forces REVIEW even
    when the model is calm (0.05)."""
    d, reason = decide(0.05, {"card_txn_count_10m":
                              settings.velocity_max_10min})
    assert d == Decision.REVIEW
    assert "velocity" in reason.lower()


def test_velocity_rule_boundary_exactly_at_limit():
    d, _ = decide(0.05, {"card_txn_count_10m":
                         settings.velocity_max_10min - 1})
    assert d == Decision.APPROVE


def test_approve_is_the_default_quiet_path():
    d, reason = decide(0.01, {"card_txn_count_10m": 0})
    assert d == Decision.APPROVE


def test_reason_codes_are_audit_friendly():
    """Every non-APPROVE decision must carry a machine-readable reason
    (regulatory/audit requirement)."""
    cases = [(settings.block_threshold + 0.05, {"card_txn_count_10m": 0}),
             (settings.review_threshold + 0.05, {"card_txn_count_10m": 0}),
             (0.05, {"card_txn_count_10m": settings.velocity_max_10min})]
    for p, f in cases:
        d, reason = decide(p, f)
        if d != Decision.APPROVE:
            assert isinstance(reason, str) and len(reason) > 5


def test_missing_velocity_feature_defaults_to_zero():
    """Defensive: absent feature must not crash the engine (treated as 0)."""
    d, _ = decide(0.10, {})
    assert d == Decision.APPROVE


def test_none_velocity_feature_defaults_to_zero():
    d, _ = decide(0.10, {"card_txn_count_10m": None})
    assert d == Decision.APPROVE
