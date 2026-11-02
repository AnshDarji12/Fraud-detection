"""Central configuration - everything driven by environment variables."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "localhost:29092")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    postgres_dsn: str = os.getenv(
        "POSTGRES_DSN", "postgresql+psycopg2://fraud@localhost:5433/fraud"
    )
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    inference_url: str = os.getenv("INFERENCE_URL", "http://localhost:8000")

    model_name: str = os.getenv("MODEL_NAME", "fraud_xgb")
    model_stage: str = os.getenv("MODEL_STAGE", "Production")
    model_path: str = os.getenv("MODEL_PATH", "artifacts/models/fraud_xgb.joblib")

    block_threshold: float = _f("BLOCK_THRESHOLD", 0.85)
    review_threshold: float = _f("REVIEW_THRESHOLD", 0.50)
    velocity_max_10min: int = _i("VELOCITY_RULE_MAX_TXNS_10MIN", 5)

    label_delay_seconds: int = _i("LABEL_DELAY_SECONDS", 30)


settings = Settings()
