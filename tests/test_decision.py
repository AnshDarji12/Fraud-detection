from src.inference.app import decide


def test_block_on_high_score():
    d, reason = decide(0.93, {"card_txn_count_10m": 1})
    assert d.value == "BLOCK"
    assert "block" in reason


def test_review_on_mid_score():
    d, _ = decide(0.60, {"card_txn_count_10m": 1})
    assert d.value == "REVIEW"


def test_approve_on_low_score():
    d, _ = decide(0.10, {"card_txn_count_10m": 1})
    assert d.value == "APPROVE"


def test_velocity_rule_overrides_approve():
    # low score but card is being tested rapidly -> must NOT auto-approve
    d, reason = decide(0.05, {"card_txn_count_10m": 9})
    assert d.value == "REVIEW"
    assert "velocity" in reason


def test_velocity_boundary():
    d, _ = decide(0.05, {"card_txn_count_10m": 5})  # == max -> triggers
    assert d.value == "REVIEW"
    d, _ = decide(0.05, {"card_txn_count_10m": 2})
    assert d.value == "APPROVE"
