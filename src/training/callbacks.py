"""
Custom Keras callbacks for MLflow metric logging and model checkpointing.
"""
from __future__ import annotations

import mlflow
import tensorflow as tf
from loguru import logger


class MLflowLoggingCallback(tf.keras.callbacks.Callback):
    """Logs per-epoch metrics and learning rate to MLflow."""

    def __init__(self, log_every_n_epochs: int = 1) -> None:
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if epoch % self.log_every_n_epochs != 0:
            return
        logs = logs or {}
        metrics = {k: float(v) for k, v in logs.items()}
        # Log current learning rate
        if hasattr(self.model.optimizer, "learning_rate"):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            metrics["learning_rate"] = lr
        mlflow.log_metrics(metrics, step=epoch)

    def on_train_end(self, logs: dict | None = None) -> None:
        logger.info("Training complete — all metrics logged to MLflow.")


class EarlyStoppingWithLogging(tf.keras.callbacks.EarlyStopping):
    """EarlyStopping that also logs the best epoch to MLflow."""

    def on_train_end(self, logs: dict | None = None) -> None:
        super().on_train_end(logs)
        mlflow.log_param("best_epoch", self.best_epoch + 1)
        mlflow.log_metric("best_val_loss", float(self.best))
