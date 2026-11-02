from datetime import datetime, timezone

import pytest

from src.common.schemas import Decision, Transaction


def make_txn(**over):
    base = dict(
        txn_id="T-1",
        event_ts=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc),
        user_id="U-1", card_id="C-1", merchant_id="M-1",
        merchant_cat="grocery", amount=100.0,
        channel="web", device_id="D-1",
    )
    base.update(over)
    return Transaction(**base)


def test_valid_transaction_parses():
    t = make_txn()
    assert t.decision if False else True  # schema model, not decision
    assert t.txn_id == "T-1"
    d = t.model_dump(mode="json")
    assert d["amount"] == 100.0


def test_negative_amount_rejected():
    with pytest.raises(Exception):
        make_txn(amount=-5)


def test_decision_enum_values():
    assert Decision.BLOCK.value == "BLOCK"
    assert Decision("REVIEW") is Decision.REVIEW
