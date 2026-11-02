"""ONLINE feature store - sliding-window behavioural features.

Two interchangeable backends with IDENTICAL semantics:
  - RedisFeatureStore : production path (Redis ZSET sliding windows)
  - InMemoryFeatureStore : fallback for tests / no-infra runs

Both implement compute_and_update(txn) -> feature dict where features
are computed from PRIOR history, then the current event is written
(matching the offline `closed='left'` contract).
"""
import time
from typing import Dict, List

from src.common.features import (DEFAULT_TIME_SINCE_PREV_SEC, base_features)

MIN10 = 600.0
HOUR = 3600.0
DAY = 24 * HOUR
WEEK = 7 * DAY


class InMemoryFeatureStore:
    """Pure-python sliding windows - used in tests and offline demos.

    Parity contract (pinned by tests/model/test_offline_online_parity.py):
    - window features count PRIOR events only: 0 < (now - ts) <= window
      (same-timestamp peers and future-dated events are excluded,
       matching the offline pandas rolling closed='left' semantics)
    - card_time_since_prev_sec / device/merchant-changed use the card's
      LAST SEEN event regardless of window age (no 7-day capping -
      regression fix for DEFECT-01 training/serving skew)
    """

    def __init__(self):
        # card_id -> list of dicts(ts, amount, merchant_id, device_id)
        self._card: Dict[str, List[dict]] = {}
        self._user: Dict[str, List[dict]] = {}
        # card_id -> (ts, device_id, merchant_id) of last event, NEVER pruned
        self._card_last: Dict[str, tuple] = {}

    @staticmethod
    def _prune(events: List[dict], now: float, window: float) -> List[dict]:
        cut = now - window
        return [e for e in events if e["ts"] >= cut]

    def compute_and_update(self, t) -> Dict[str, float]:
        now = t.event_ts.timestamp()
        card_events = self._prune(self._card.get(t.card_id, []), now, WEEK)
        user_events = self._prune(self._user.get(t.user_id, []), now, WEEK)

        f = base_features(t.amount, t.event_ts.hour)
        f["card_txn_count_10m"] = float(
            sum(1 for e in card_events if 0 < now - e["ts"] <= MIN10))
        f["card_txn_count_1h"] = float(
            sum(1 for e in card_events if 0 < now - e["ts"] <= HOUR))
        f["card_amount_sum_1h"] = float(
            sum(e["amount"] for e in card_events
                if 0 < now - e["ts"] <= HOUR))
        f["user_txn_count_24h"] = float(
            sum(1 for e in user_events if 0 < now - e["ts"] <= DAY))
        prev7 = [e["amount"] for e in user_events
                 if 0 < now - e["ts"] <= WEEK]
        f["user_amount_avg_7d"] = float(sum(prev7) / len(prev7)) if prev7 else 0.0

        last = self._card_last.get(t.card_id)
        if last:
            delta = max(0.0, now - last[0])
            f["card_time_since_prev_sec"] = delta
            f["card_device_changed"] = float(
                last[1] != t.device_id and delta <= DAY)
            f["card_merchant_changed"] = float(
                last[2] != t.merchant_id and delta <= HOUR)
        else:
            f["card_time_since_prev_sec"] = DEFAULT_TIME_SINCE_PREV_SEC
            f["card_device_changed"] = 0.0
            f["card_merchant_changed"] = 0.0

        # ---- update state with the CURRENT txn ----
        card_events.append({"ts": now, "amount": t.amount,
                            "merchant_id": t.merchant_id,
                            "device_id": t.device_id})
        user_events.append({"ts": now, "amount": t.amount})
        self._card[t.card_id] = self._prune(card_events, now, WEEK)
        self._user[t.user_id] = self._prune(user_events, now, WEEK)
        self._card_last[t.card_id] = (now, t.device_id, t.merchant_id)
        return f


class RedisFeatureStore:
    """Redis ZSET sliding windows.

    card:{id}:txns  member = "{txn_id}|{amount}|{merchant}|{device}" score = ts_ms
    user:{id}:txns  member = "{txn_id}|{amount}"                  score = ts_ms
    card:{id}:last  string "ts_ms|device|merchant" (never TTL-pruned short)

    Window reads use score range [now-window, now-1] so that
    same-timestamp peers are excluded exactly like the offline builder
    (second-resolution event contract).
    """

    def __init__(self, redis_url: str):
        import redis as redis_lib
        self.r = redis_lib.Redis.from_url(redis_url, decode_responses=True)
        self.r.ping()

    @staticmethod
    def _member(t) -> str:
        return f"{t.txn_id}|{t.amount}|{t.merchant_id}|{t.device_id}"

    def compute_and_update(self, t) -> Dict[str, float]:
        r = self.r
        now_ms = t.event_ts.timestamp() * 1000.0
        card_key = f"card:{t.card_id}:txns"
        user_key = f"user:{t.user_id}:txns"
        last_key = f"card:{t.card_id}:last"

        f = base_features(t.amount, t.event_ts.hour)

        # [now-W, now-1]: includes exact window boundary, excludes same-ts
        def _range(window: float):
            return (now_ms - window * 1000.0, now_ms - 1.0)

        lo10, hi = _range(MIN10)
        f["card_txn_count_10m"] = float(r.zcount(card_key, lo10, hi) or 0)
        lo1h, _ = _range(HOUR)
        f["card_txn_count_1h"] = float(r.zcount(card_key, lo1h, hi) or 0)

        sum_1h = 0.0
        for m in r.zrangebyscore(card_key, lo1h, hi):
            try:
                sum_1h += float(m.split("|")[1])
            except (IndexError, ValueError):
                continue
        f["card_amount_sum_1h"] = sum_1h

        lo24h, _ = _range(DAY)
        f["user_txn_count_24h"] = float(
            r.zcount(user_key, lo24h, hi) or 0)

        lo7d, _ = _range(WEEK)
        amounts7 = [float(m.split("|")[1])
                    for m in r.zrangebyscore(user_key, lo7d, hi)]
        f["user_amount_avg_7d"] = (sum(amounts7) / len(amounts7)) if amounts7 else 0.0

        raw = r.get(last_key)
        if raw:
            parts = str(raw).split("|")
            prev_ts_ms = float(parts[0])
            prev_device = parts[1] if len(parts) > 1 else ""
            prev_merchant = parts[2] if len(parts) > 2 else ""
            delta_sec = max(0.0, (now_ms - prev_ts_ms) / 1000.0)
            f["card_time_since_prev_sec"] = delta_sec
            f["card_device_changed"] = float(
                prev_device != t.device_id and delta_sec <= DAY)
            f["card_merchant_changed"] = float(
                prev_merchant != t.merchant_id and delta_sec <= HOUR)
        else:
            f["card_time_since_prev_sec"] = DEFAULT_TIME_SINCE_PREV_SEC
            f["card_device_changed"] = 0.0
            f["card_merchant_changed"] = 0.0

        # ---- write current txn ----
        pipe = r.pipeline()
        pipe.zadd(card_key, {self._member(t): now_ms})
        pipe.zremrangebyscore(card_key, "-inf", now_ms - WEEK * 1000)
        pipe.expire(card_key, int(WEEK) + 3600)
        pipe.zadd(user_key, {f"{t.txn_id}|{t.amount}": now_ms})
        pipe.zremrangebyscore(user_key, "-inf", now_ms - WEEK * 1000)
        pipe.expire(user_key, int(WEEK) + 3600)
        pipe.set(last_key, f"{now_ms}|{t.device_id}|{t.merchant_id}")
        pipe.execute()
        return f


def make_store(redis_url: str):
    """Prefer Redis; fall back to in-memory if Redis is unreachable."""
    if redis_url.startswith("redis"):
        try:
            store = RedisFeatureStore(redis_url)
            print("[ok] online feature store: Redis")
            return store
        except Exception as e:
            print(f"[warn] Redis unavailable ({e.__class__.__name__}) - "
                  f"using in-memory feature store")
    else:
        print("[warn] REDIS_URL not redis:// - using in-memory feature store")
    return InMemoryFeatureStore()
