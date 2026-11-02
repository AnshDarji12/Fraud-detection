"""Pydantic data contracts + Kafka topic names (the 'data contract' layer)."""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ---------------- Kafka topics ----------------
TOPIC_RAW = "txn.raw"
TOPIC_SCORED = "txn.scored"
TOPIC_ALERTS = "txn.alerts"
TOPIC_LABELS = "txn.labels"
TOPIC_DLQ = "txn.dlq"


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class Transaction(BaseModel):
    """One payment event flowing through the platform."""

    txn_id: str
    event_ts: datetime
    user_id: str
    card_id: str
    merchant_id: str
    merchant_cat: str
    amount: float = Field(gt=0)
    currency: str = "USD"
    channel: str = "online"
    device_id: str = ""
    ip: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None


class ScoredTransaction(Transaction):
    p_fraud: float
    decision: Decision
    reason: str = ""
    model_version: str = ""
