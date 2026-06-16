"""
Main training entry point — orchestrates data loading, feature engineering,
model training, evaluation, and MLflow experiment tracking / model registration.

Usage:
    python src/training/train.py
    python src/training/train.py experiment=cf_experiment
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import hydra
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.ingestion import download_movielens, load_movies, load_ratings, load_users
from data.preprocessing import DataPreprocessor
from evaluation.metrics import compute_metrics, compute_ranking_metrics
from features.feature_engineering import FeatureEngineer
from models import HybridRecommender, NeuralCollaborativeFiltering, TwoTowerModel
from training.callbacks import EarlyStoppingWithLogging, MLflowLoggingCallback
from training.dataset import build_tf_dataset


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Reproducibility ───────────────────────────────────────────────────────
    tf.random.set_seed(cfg.data.random_seed)
    np.random.seed(cfg.data.random_seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading data ...")
    data_dir = download_movielens(dataset="1m", raw_dir=cfg.data.raw_path)
    ratings = load_ratings(data_dir, dataset="1m")
    movies = load_movies(data_dir, dataset="1m")
    users = load_users(data_dir, dataset="1m")

    preprocessor = DataPreprocessor(
        min_ratings_per_user=cfg.data.min_ratings_per_user,
        min_ratings_per_movie=cfg.data.min_ratings_per_movie,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        random_seed=cfg.data.random_seed,
    )
    train_df, val_df, test_df = preprocessor.fit_transform(ratings, movies, users)
    preprocessor.save(cfg.data.processed_path)

    # ── Feature Engineering ───────────────────────────────────────────────────
    feat_eng = FeatureEngineer()
    feat_eng.fit_transform(train_df, movies)
    feat_eng.save(cfg.data.features_path)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = build_tf_dataset(
        train_df,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_ds = build_tf_dataset(val_df, batch_size=cfg.training.batch_size, shuffle=False)
    test_ds = build_tf_dataset(test_df, batch_size=cfg.training.batch_size, shuffle=False)

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.experiment.name) as run:
        # Log config as params
        mlflow.log_params(
            {
                "model_type": cfg.model.type,
                "embedding_dim": cfg.model.user_embedding_dim,
                "hidden_layers": str(cfg.model.hidden_layers),
                "dropout_rate": cfg.model.dropout_rate,
                "batch_size": cfg.training.batch_size,
                "learning_rate": cfg.training.learning_rate,
                "epochs": cfg.training.epochs,
                "n_users": preprocessor.n_users,
                "n_movies": preprocessor.n_movies,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
            }
        )
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.json")
        mlflow.set_tags(
            {
                "experiment": cfg.experiment.name,
                "dataset": "movielens_1m",
                **cfg.experiment.get("tags", {}),
            }
        )

        # ── Build model ───────────────────────────────────────────────────────
        model = _build_model(cfg, preprocessor.n_users, preprocessor.n_movies)
        logger.info(f"Built {cfg.model.type} model.")

        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        )

        # ── Callbacks ─────────────────────────────────────────────────────────
        checkpoint_path = Path("artifacts") / run.info.run_id / "best_model"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        callbacks = [
            MLflowLoggingCallback(log_every_n_epochs=1),
            EarlyStoppingWithLogging(
                monitor="val_loss",
                patience=cfg.training.early_stopping.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path / "weights.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            ),
        ]

        # ── Train ─────────────────────────────────────────────────────────────
        logger.info("Starting training ...")
        start = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.training.epochs,
            callbacks=callbacks,
            verbose=2,
        )
        train_duration = time.time() - start
        mlflow.log_metric("train_duration_seconds", train_duration)

        # ── Evaluate ──────────────────────────────────────────────────────────
        logger.info("Evaluating on test set ...")
        test_results = model.evaluate(test_ds, verbose=0, return_dict=True)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_results.items()})

        # Ranking metrics (NDCG@10, Precision@10, Recall@10)
        ranking = compute_ranking_metrics(model, test_df, k=10)
        mlflow.log_metrics(ranking)
        logger.info(f"Test results: {test_results} | Ranking: {ranking}")

        # ── Log model to registry ─────────────────────────────────────────────
        logger.info("Logging model to MLflow registry ...")
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=cfg.mlflow.model_name,
            pip_requirements=["tensorflow==2.15.0", "mlflow==2.9.2"],
        )

        # Log preprocessor artifacts
        mlflow.log_artifacts(cfg.data.processed_path, artifact_path="preprocessor")

        logger.info(
            f"Run {run.info.run_id} complete. "
            f"Test RMSE: {test_results.get('rmse', 'N/A'):.4f}"
        )

    logger.info("Training pipeline finished successfully.")


def _build_model(
    cfg: DictConfig,
    n_users: int,
    n_movies: int,
) -> tf.keras.Model:
    model_type = cfg.model.type
    kwargs = dict(
        n_users=n_users,
        n_items=n_movies,
        embedding_dim=cfg.model.user_embedding_dim,
        hidden_layers=list(cfg.model.hidden_layers),
        dropout_rate=cfg.model.dropout_rate,
        l2=cfg.model.l2_regularization,
    )
    if model_type == "two_tower":
        return TwoTowerModel(**kwargs)
    if model_type == "ncf":
        return NeuralCollaborativeFiltering(**kwargs)
    if model_type == "hybrid":
        # Hybrid needs feature dimensions; use placeholder values here
        return HybridRecommender(**kwargs, n_user_features=20, n_item_features=30)
    raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()
