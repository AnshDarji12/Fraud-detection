"""SINGLE SOURCE OF TRUTH for the model feature set.

The offline trainer (src/training/features.py) and the online feature
store (src/streaming/features_online.py) MUST both produce exactly
these keys with the same semantics. Divergence between offline and
online features is the classic fraud-ML production bug - this module
prevents it.

Semantics (important):
- "count" features EXCLUDE the current transaction (computed from
  prior history only) - matching online scoring where we score BEFORE
  writing the current event into the store.
"""

from typing import Dict, List
import math

FEATURES: List[str] = [
    "amount",
    "log_amount",
    "hour_of_day",
    "is_night",
    "card_txn_count_10m",
    "card_txn_count_1h",
    "card_amount_sum_1h",
    "user_txn_count_24h",
    "user_amount_avg_7d",
    "card_time_since_prev_sec",
    "card_device_changed",
    "card_merchant_changed",
]

DEFAULT_TIME_SINCE_PREV_SEC = 24 * 3600.0


def base_features(amount: float, hour_of_day: int) -> Dict[str, float]:
    """Features derivable from the transaction alone (no history)."""
    return {
        "amount": float(amount),
        "log_amount": math.log1p(float(amount)),
        "hour_of_day": int(hour_of_day) % 24,
        "is_night": 1.0 if (hour_of_day >= 22 or hour_of_day < 6) else 0.0,
    }


def to_vector(feature_dict: Dict[str, float]) -> List[float]:
    """Order-stable vector for the model - MUST follow FEATURES order."""
    return [float(feature_dict[name]) for name in FEATURES]
