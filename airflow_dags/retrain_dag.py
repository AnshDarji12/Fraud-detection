"""Airflow DAG - automated retraining pipeline (optional component).

Airflow is free/open-source. Run it with the official docker-compose
(https://airflow.apache.org/docs/apache-airflow/stable/howto/
docker-compose/index.html) and drop this file into AIRFLOW__CORE__dags_folder.

If you do not want Airflow yet, the same job runs via `scripts/retrain.sh`
or a plain cron entry - the platform works without it.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraud_model_retraining",
    description="Nightly: regenerate data -> retrain -> register in MLflow",
    schedule="0 3 * * *",  # 03:00 daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["fraud", "mlops"],
) as dag:
    generate_data = BashOperator(
        task_id="generate_training_data",
        bash_command="cd /app && python -m src.datagen.synth_generator "
                     "--rows 20000 --out data/processed/demo_transactions.csv",
    )

    train_model = BashOperator(
        task_id="train_and_register",
        bash_command="cd /app && python -m src.training.train "
                     "--csv data/processed/demo_transactions.csv",
    )

    generate_data >> train_model
