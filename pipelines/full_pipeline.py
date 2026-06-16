"""
End-to-end ML pipeline:
  1. Data ingestion
  2. Preprocessing
  3. Feature engineering
  4. Model training (with MLflow tracking)
  5. Evaluation
  6. Model promotion (registry)
  7. Deployment health check

Run:
    python pipelines/full_pipeline.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from data.ingestion import download_movielens, load_movies, load_ratings, load_users
from data.preprocessing import DataPreprocessor
from features.feature_engineering import FeatureEngineer
from training.dataset import build_tf_dataset
from evaluation.metrics import compute_ranking_metrics


def run_pipeline(
    dataset: str = "1m",
    model_type: str = "two_tower",
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    embedding_dim: int = 64,
    experiment_name: str = "cinerecops",
    model_name: str = "cinerecops-recommender",
    tracking_uri: str = "http://localhost:5000",
    promote: bool = True,
) -> str:
    """Run the full training pipeline end-to-end.

    Returns:
        MLflow run ID of the completed training run.
    """
    import mlflow
    import mlflow.tensorflow
    import tensorflow as tf
    import numpy as np
    import time

    from models import TwoTowerModel, NeuralCollaborativeFiltering

    logger.info("=" * 60)
    logger.info(f"Starting CineRecOps pipeline — model={model_type}, epochs={epochs}")
    logger.info("=" * 60)

    # ── 1. Ingest ──────────────────────────────────────────────────────────────
    logger.info("Step 1/6: Data ingestion ...")
    data_dir = download_movielens(dataset=dataset, raw_dir="data/raw")
    ratings = load_ratings(data_dir, dataset=dataset)
    movies = load_movies(data_dir, dataset=dataset)
    users = load_users(data_dir, dataset=dataset)

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    logger.info("Step 2/6: Preprocessing ...")
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.fit_transform(ratings, movies, users)
    preprocessor.save("data/processed")

    # Save movies for serving
    import pandas as pd
    movies.to_parquet("data/processed/movies.parquet", index=False)

    # ── 3. Feature engineering ─────────────────────────────────────────────────
    logger.info("Step 3/6: Feature engineering ...")
    feat_eng = FeatureEngineer()
    feat_eng.fit_transform(train_df, movies)
    feat_eng.save("data/features")

    # ── 4. Train ───────────────────────────────────────────────────────────────
    logger.info("Step 4/6: Training ...")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    train_ds = build_tf_dataset(train_df, batch_size=batch_size)
    val_ds = build_tf_dataset(val_df, batch_size=batch_size, shuffle=False)
    test_ds = build_tf_dataset(test_df, batch_size=batch_size, shuffle=False)

    if model_type == "ncf":
        model = NeuralCollaborativeFiltering(
            n_users=preprocessor.n_users,
            n_items=preprocessor.n_movies,
            embedding_dim=embedding_dim,
        )
    else:
        model = TwoTowerModel(
            n_users=preprocessor.n_users,
            n_items=preprocessor.n_movies,
            embedding_dim=embedding_dim,
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mean_squared_error",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    with mlflow.start_run(run_name=f"{model_type}_pipeline") as run:
        mlflow.log_params({
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "embedding_dim": embedding_dim,
            "n_users": preprocessor.n_users,
            "n_movies": preprocessor.n_movies,
        })

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]

        start_time = time.time()
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)
        mlflow.log_metric("train_duration_seconds", time.time() - start_time)

        # ── 5. Evaluate ────────────────────────────────────────────────────────
        logger.info("Step 5/6: Evaluation ...")
        test_results = model.evaluate(test_ds, verbose=0, return_dict=True)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_results.items()})
        ranking = compute_ranking_metrics(model, test_df, k=10)
        mlflow.log_metrics(ranking)
        logger.info(f"Test: {test_results} | Ranking: {ranking}")

        # Log model
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        mlflow.log_artifacts("data/processed", artifact_path="preprocessor")

        run_id = run.info.run_id
        logger.info(f"Training run completed: {run_id}")

    # ── 6. Promote ─────────────────────────────────────────────────────────────
    if promote:
        logger.info("Step 6/6: Promoting best model to Production ...")
        from mlflow.registry import promote_best_model  # local module
        promote_best_model(
            experiment_name=experiment_name,
            model_name=model_name,
            metric="test_rmse",
            tracking_uri=tracking_uri,
        )

    logger.info("Pipeline complete!")
    return run_id


if __name__ == "__main__":
    run_pipeline()
