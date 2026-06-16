"""
MLflow server configuration and experiment setup.
Run this once to initialize experiments in a fresh MLflow server.
"""
import os
import mlflow
from loguru import logger


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = "cinerecops",
    model_name: str = "cinerecops-recommender",
) -> None:
    """Initialize MLflow tracking server, experiments, and registered model."""
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Create experiments
    experiments = [
        (experiment_name, {"project": "cinerecops", "team": "mlops"}),
        (f"{experiment_name}_staging", {"project": "cinerecops", "env": "staging"}),
    ]
    for exp_name, tags in experiments:
        existing = client.get_experiment_by_name(exp_name)
        if existing is None:
            exp_id = client.create_experiment(exp_name, tags=tags)
            logger.info(f"Created experiment '{exp_name}' (id={exp_id}).")
        else:
            logger.info(f"Experiment '{exp_name}' already exists (id={existing.experiment_id}).")

    # Create registered model if it doesn't exist
    try:
        client.create_registered_model(
            model_name,
            description="Production movie recommendation model (TwoTower / NCF / Hybrid)",
            tags={"project": "cinerecops", "framework": "tensorflow"},
        )
        logger.info(f"Registered model '{model_name}' created.")
    except mlflow.exceptions.MlflowException:
        logger.info(f"Registered model '{model_name}' already exists.")


if __name__ == "__main__":
    setup_mlflow()
