"""
MLflow model loader — fetches the latest Production model from the registry
and holds it as a singleton for the API to use.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from loguru import logger


class ModelLoader:
    """Thread-safe singleton that loads and caches the production model."""

    _instance: "ModelLoader | None" = None

    def __init__(self) -> None:
        self._model: tf.keras.Model | None = None
        self._model_version: str = "unknown"
        self._loaded_at: float = 0.0

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(
        self,
        model_name: str | None = None,
        stage: str = "Production",
        tracking_uri: str | None = None,
    ) -> None:
        """Load model from MLflow registry or fallback to local artifacts."""
        model_name = model_name or os.getenv("MLFLOW_MODEL_NAME", "cinerecops-recommender")
        stage = stage or os.getenv("MODEL_STAGE", "Production")
        tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

        mlflow.set_tracking_uri(tracking_uri)

        try:
            model_uri = f"models:/{model_name}/{stage}"
            logger.info(f"Loading model from registry: {model_uri}")
            self._model = mlflow.tensorflow.load_model(model_uri)
            # Get version info
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[stage])
            self._model_version = versions[0].version if versions else "unknown"
            self._loaded_at = time.time()
            logger.info(f"Model v{self._model_version} loaded successfully.")
        except Exception as e:
            logger.warning(f"Registry load failed ({e}). Trying local fallback ...")
            self._load_local_fallback()

    def _load_local_fallback(self) -> None:
        """Attempt to load from a local mlruns directory."""
        local_model_dir = Path("artifacts") / "latest_model"
        if local_model_dir.exists():
            self._model = tf.keras.models.load_model(str(local_model_dir))
            self._model_version = "local"
            self._loaded_at = time.time()
            logger.info("Loaded model from local fallback.")
        else:
            logger.error("No local model found. API will return errors until a model is available.")

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def version(self) -> str:
        return self._model_version

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        user_idxs: np.ndarray,
        item_idxs: np.ndarray,
    ) -> np.ndarray:
        """Run inference with the loaded model."""
        return self.model.predict(
            (user_idxs.astype(np.int32), item_idxs.astype(np.int32)),
            batch_size=512,
            verbose=0,
        ).flatten()
