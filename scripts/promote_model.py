"""Script to promote the best model from a completed experiment to Production."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlflow"))

from registry import list_model_versions, promote_best_model
from loguru import logger

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "cinerecops")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "cinerecops-recommender")


def main():
    logger.info(f"Tracking URI: {TRACKING_URI}")
    logger.info("Current model versions:")
    list_model_versions(MODEL_NAME, tracking_uri=TRACKING_URI)

    version = promote_best_model(
        experiment_name=EXPERIMENT,
        model_name=MODEL_NAME,
        metric="test_rmse",
        lower_is_better=True,
        target_stage="Production",
        tracking_uri=TRACKING_URI,
    )
    if version:
        logger.info(f"Successfully promoted version {version} to Production.")
    else:
        logger.error("Promotion failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
