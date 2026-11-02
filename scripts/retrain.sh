#!/usr/bin/env bash
# Automated retraining job - used by cron OR the Airflow DAG.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] generating fresh training data ..."
python -m src.datagen.synth_generator --rows 20000 \
  --out data/processed/demo_transactions.csv

echo "[2/3] training + registering model ..."
python -m src.training.train --csv data/processed/demo_transactions.csv

echo "[3/3] done. MLflow registry now serves the new Production version."
echo "      Restart the inference service to pick it up: docker compose restart inference"
