"""Model loading with graceful degradation.

Priority:
  1. MLflow registry  models:/{MODEL_NAME}/{MODEL_STAGE}   (enterprise path)
  2. local joblib artifact  (lets the API boot before MLflow/registry)
"""
import os

from src.common.config import settings


class ModelBundle:
    def __init__(self, model, version: str):
        self.model = model
        self.version = version

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def load_model() -> ModelBundle:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow_uri)
        model = mlflow.sklearn.load_model(
            f"models:/{settings.model_name}/{settings.model_stage}")
        client = MlflowClient(tracking_uri=settings.mlflow_uri)
        versions = client.get_latest_versions(settings.model_name,
                                              [settings.model_stage])
        version = f"mlflow:v{versions[0].version}" if versions else "mlflow"
        print(f"[ok] model loaded from MLflow registry ({version})")
        return ModelBundle(model, version)
    except Exception as e:
        print(f"[warn] MLflow model unavailable ({e.__class__.__name__}: "
              f"{str(e)[:120]}) - trying local artifact")

    path = settings.model_path
    if os.path.exists(path):
        import joblib
        model = joblib.load(path)
        print(f"[ok] model loaded from local artifact {path}")
        return ModelBundle(model, "local")

    raise FileNotFoundError(
        f"No model found: registry models:/{settings.model_name}/"
        f"{settings.model_stage} unreachable and '{path}' missing. "
        f"Run `make train` first (or `make train` with MLflow up).")
