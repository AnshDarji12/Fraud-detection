"""Kafka replay producer - turns a static CSV into a LIVE stream.

Simulates enterprise ingestion:
  - publishes transactions to `txn.raw` at a controlled rate (keyed by
    card_id so Kafka keeps per-card ordering for feature computation)
  - labels are published SEPARATELY to `txn.labels` after a configurable
    real-world delay (fraud labels arrive late, like real chargebacks)

This is what makes the project genuinely "real-time" for demos.
"""
import argparse
import heapq
import json
import threading
import time

import pandas as pd

from src.common.config import settings
from src.common.schemas import TOPIC_LABELS, TOPIC_RAW, Transaction

try:
    from kafka import KafkaProducer
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install kafka-python-ng  (see requirements.txt)") from e


def json_serializer(v) -> bytes:
    return json.dumps(v, default=str).encode("utf-8")


class LabelScheduler:
    """Heap-based scheduler that publishes labels with a simulated delay."""

    def __init__(self, producer, delay_seconds: float):
        self.producer = producer
        self.delay = delay_seconds
        self.heap: list = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def schedule(self, send_at: float, txn_id: str, is_fraud: int):
        with self.lock:
            heapq.heappush(self.heap, (send_at, txn_id, is_fraud))

    def _run(self):
        while not self.stop.is_set():
            with self.lock:
                due = self.heap[:1]
            if due and due[0][0] <= time.time():
                with self.lock:
                    _, txn_id, is_fraud = heapq.heappop(self.heap)
                self.producer.send(TOPIC_LABELS, {
                    "txn_id": txn_id,
                    "is_fraud": int(is_fraud),
                    "label_ts": time.time(),
                })
            else:
                time.sleep(0.2)

    def close(self):
        self.stop.set()
        self.worker.join(timeout=2)


def main():
    ap = argparse.ArgumentParser(description="Kafka transaction replay producer")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rate", type=float, default=20, help="transactions / second")
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["event_ts"])
    df = df.sort_values("event_ts").reset_index(drop=True)
    print(f"[ok] loaded {len(df)} transactions from {args.csv} "
          f"({df['is_fraud'].mean() * 100:.2f}% fraud)")

    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        key_serializer=str.encode,
        value_serializer=json_serializer,
        linger_ms=5,
    )
    labels = LabelScheduler(producer, settings.label_delay_seconds)

    interval = 1.0 / max(args.rate, 0.1)
    sent = 0
    for rnd in range(args.rounds):
        t0 = time.time()
        for i, row in enumerate(df.itertuples(index=False)):
            txn = Transaction(
                txn_id=row.txn_id,
                event_ts=row.event_ts.to_pydatetime(),
                user_id=row.user_id,
                card_id=row.card_id,
                merchant_id=row.merchant_id,
                merchant_cat=row.merchant_cat,
                amount=float(row.amount),
                currency=row.currency,
                channel=row.channel,
                device_id=row.device_id,
                ip=row.ip,
                lat=float(row.lat) if pd.notna(row.lat) else None,
                lon=float(row.lon) if pd.notna(row.lon) else None,
            )
            send_time = time.time()
            producer.send(TOPIC_RAW, key=txn.card_id,
                          value=txn.model_dump(mode="json"))
            labels.schedule(send_time + settings.label_delay_seconds,
                            txn.txn_id, int(row.is_fraud))
            sent += 1
            if sent % (int(args.rate) * 20 or 100) == 0:
                producer.flush()
                print(f"[..] sent {sent} txns "
                      f"(round {rnd + 1}/{args.rounds}, row {i + 1}/{len(df)})")
            # pacing relative to round start keeps cumulative rate exact
            target = t0 + (rnd * len(df) + i + 1) * interval
            sleep_for = target - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

    producer.flush()
    print(f"[ok] finished sending {sent} transactions; flushing labels "
          f"for {settings.label_delay_seconds}s ...")
    time.sleep(settings.label_delay_seconds)
    labels.close()
    producer.close()


if __name__ == "__main__":
    main()
