"""
MLflow Model Registry helpers — promote, compare, archive, and serve models.
"""
from __future__ import annotations

import os
from typing import Optional

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient


def get_client(tracking_uri: str | None = None) -> MlflowClient:
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def promote_best_model(
    experiment_name: str,
    model_name: str,
    metric: str = "test_rmse",
    lower_is_better: bool = True,
    target_stage: str = "Production",
    tracking_uri: str | None = None,
) -> Optional[str]:
    """Find the best run in an experiment and promote its model to Production.

    Args:
        experiment_name: MLflow experiment to search.
        model_name: Registered model name in the registry.
        metric: Metric to rank runs by.
        lower_is_better: True for loss/error metrics, False for accuracy metrics.
        target_stage: Registry stage to promote to (Staging, Production).
        tracking_uri: Optional override for tracking server URL.

    Returns:
        Version string of the promoted model, or None if no suitable run found.
    """
    client = get_client(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error(f"Experiment '{experiment_name}' not found.")
        return None

    # Find best run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric} > 0",
        order_by=[f"metrics.{metric} {'ASC' if lower_is_better else 'DESC'}"],
        max_results=1,
    )
    if not runs:
        logger.error(f"No runs with metric '{metric}' found in '{experiment_name}'.")
        return None

    best_run = runs[0]
    best_metric = best_run.data.metrics.get(metric, "N/A")
    logger.info(f"Best run: {best_run.info.run_id} | {metric}={best_metric}")

    # Get the model version registered from that run
    versions = client.search_model_versions(f"name='{model_name}'")
    run_versions = [v for v in versions if v.run_id == best_run.info.run_id]

    if not run_versions:
        logger.error(f"No registered model version found for run {best_run.info.run_id}.")
        return None

    version_to_promote = run_versions[0].version

    # Archive current Production models
    current_prod = client.get_latest_versions(model_name, stages=[target_stage])
    for prod_version in current_prod:
        if prod_version.version != version_to_promote:
            client.transition_model_version_stage(
                name=model_name,
                version=prod_version.version,
                stage="Archived",
            )
            logger.info(f"Archived model version {prod_version.version}.")

    # Promote best model
    client.transition_model_version_stage(
        name=model_name,
        version=version_to_promote,
        stage=target_stage,
        archive_existing_versions=False,
    )
    logger.info(
        f"Promoted model '{model_name}' v{version_to_promote} to {target_stage}. "
        f"({metric}={best_metric})"
    )
    return version_to_promote


def list_model_versions(model_name: str, tracking_uri: str | None = None) -> None:
    """Print a table of all versions and their stages for a registered model."""
    client = get_client(tracking_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logger.info(f"No versions found for model '{model_name}'.")
        return

    logger.info(f"\n{'Version':<10} {'Stage':<15} {'Run ID':<36} {'Status'}")
    logger.info("-" * 75)
    for v in sorted(versions, key=lambda x: int(x.version)):
        logger.info(f"{v.version:<10} {v.current_stage:<15} {v.run_id:<36} {v.status}")


def set_model_alias(
    model_name: str,
    version: str,
    alias: str = "champion",
    tracking_uri: str | None = None,
) -> None:
    """Set an alias (e.g., 'champion') on a model version for easy loading."""
    client = get_client(tracking_uri)
    client.set_registered_model_alias(model_name, alias, version)
    logger.info(f"Set alias '{alias}' on {model_name} v{version}.")
