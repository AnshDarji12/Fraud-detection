"""Bounded LRU dedup for at-least-once delivery (Kafka default).

Regression fix for DEFECT-02: redelivered transactions must be scored
and alerted EXACTLY ONCE. Kafka guarantees at-least-once, so the
processor - not the broker - owns idempotency.
"""
from collections import OrderedDict
from typing import Dict


class TxnDeduper:
    """Remembers the last `maxlen` txn_ids; O(1) membership checks."""

    def __init__(self, maxlen: int = 100_000):
        self._seen: Dict[str, None] = OrderedDict()
        self._maxlen = max(1, int(maxlen))
        self.hits = 0

    def seen_before(self, txn_id: str) -> bool:
        if txn_id in self._seen:
            self._seen.move_to_end(txn_id)
            self.hits += 1
            return True
        self._seen[txn_id] = None
        if len(self._seen) > self._maxlen:
            self._seen.popitem(last=False)
        return False

    def __len__(self) -> int:
        return len(self._seen)
