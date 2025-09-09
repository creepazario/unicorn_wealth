"""Model Registry management for MLflow.

This module provides ModelRegistryManager to:
- Register a trained model ensemble (1h, 4h, 8h) from a parent MLflow run.
- Promote the entire ensemble from Staging to Production atomically (all-or-nothing).

Implementation assumptions based on ModelTrainer:
- Parent training run contains three nested child runs named "{horizon}_model"
  for horizons "1h", "4h", and "8h".
- Each child run logs a model artifact at artifact_path="model".
- Thus, the logged model artifact URI for a child run is "runs:/{run_id}/model".
"""

from __future__ import annotations

from typing import Dict, List, Mapping

import logging

import mlflow
from mlflow.tracking import MlflowClient

__all__ = ["ModelRegistryManager"]

logger = logging.getLogger(__name__)


_HORIZONS: tuple[str, ...] = ("1h", "4h", "8h")
_MODEL_NAMES: Mapping[str, str] = {
    "1h": "uw-catboost-1h",
    "4h": "uw-catboost-4h",
    "8h": "uw-catboost-8h",
}


class ModelRegistryManager:
    """Handles registration and promotion of the CatBoost model ensemble in MLflow.

    Methods
    -------
    register_models(parent_run_id: str) -> Dict[str, int]
        Register the three child-run models under unique names and return
        a mapping of model name to the new version number.

    promote_challenger_ensemble(new_versions: Dict[str, int]) -> None
        Promote the entire ensemble of new versions to Production atomically,
        archiving any current Production versions first. If any step fails,
        the method attempts to roll back to the original state and raises.
    """

    def __init__(self) -> None:
        self.client = MlflowClient()

    # -------------------- Registration --------------------
    def register_models(self, parent_run_id: str) -> Dict[str, int]:
        """Register the three horizon models found under the given parent run.

        Parameters
        ----------
        parent_run_id: str
            The MLflow run ID of the parent training run.

        Returns
        -------
        Dict[str, int]
            Mapping from model name (e.g., "uw-catboost-1h") to the new
            model version number created by the registry.
        """
        parent_run = self.client.get_run(parent_run_id)
        experiment_id = parent_run.info.experiment_id

        # Find nested child runs under the specified parent
        filter_str = f"tags.mlflow.parentRunId = '{parent_run_id}'"
        runs = self.client.search_runs([experiment_id], filter_string=filter_str)
        if not runs:
            raise ValueError(
                f"No child runs found for parent run_id={parent_run_id} in experiment {experiment_id}"
            )

        # Map horizon -> child run by run name tag "mlflow.runName" (e.g., "1h_model")
        horizon_to_run_id: Dict[str, str] = {}
        for r in runs:
            run_name = r.data.tags.get("mlflow.runName") or ""
            if run_name.endswith("_model"):
                horizon = run_name.replace("_model", "")
                if horizon in _HORIZONS:
                    horizon_to_run_id[horizon] = r.info.run_id

        missing = [h for h in _HORIZONS if h not in horizon_to_run_id]
        if missing:
            raise ValueError(
                "Missing expected child runs for horizons: " + ", ".join(missing)
            )

        # Register each child run's model artifact under its registry name
        name_to_version: Dict[str, int] = {}
        for horizon in _HORIZONS:
            child_run_id = horizon_to_run_id[horizon]
            artifact_uri = f"runs:/{child_run_id}/model"
            model_name = _MODEL_NAMES[horizon]

            logger.info(
                "Registering model for horizon=%s as name=%s from artifact_uri=%s",
                horizon,
                model_name,
                artifact_uri,
            )
            mv = mlflow.register_model(model_uri=artifact_uri, name=model_name)

            # mlflow.entities.model_registry.ModelVersion.version is typically str-like
            try:
                version_int = int(getattr(mv, "version"))
            except Exception:  # pragma: no cover - defensive
                version_int = int(str(getattr(mv, "version")))

            name_to_version[model_name] = version_int

        return name_to_version

    # -------------------- Promotion --------------------
    def promote_challenger_ensemble(self, new_versions: Dict[str, int]) -> None:
        """Promote the three model versions to Production atomically.

        The method will:
        1) Identify current Production versions for each model and archive them.
        2) Transition all new versions from Staging to Production.
        If any step fails for any model, attempt to roll back all transitions
        so that the ensemble behaves as all-or-nothing.

        Parameters
        ----------
        new_versions: Dict[str, int]
            Mapping from model name (e.g., "uw-catboost-1h") to new version
            numbers returned by `register_models`.
        """
        required_names = list(_MODEL_NAMES.values())
        # Validate presence of all three models
        for name in required_names:
            if name not in new_versions:
                raise ValueError(f"new_versions is missing required model name: {name}")

        # Pre-validate that the target new versions exist (and capture their stages)
        for name in required_names:
            ver = new_versions[name]
            _ = self.client.get_model_version(name=name, version=str(ver))

        # Collect current production versions for potential rollback
        current_prod: Dict[str, List[mlflow.entities.model_registry.ModelVersion]] = {}
        for name in required_names:
            prod = self.client.get_latest_versions(name=name, stages=["Production"])  # type: ignore[arg-type]
            current_prod[name] = list(prod or [])

        archived: List[tuple[str, str]] = []  # list of (name, version) archived
        promoted: List[tuple[str, str]] = []  # list of (name, version) promoted

        try:
            # Phase 1: archive current production versions (if any)
            for name in required_names:
                for mv in current_prod[name]:
                    self.client.transition_model_version_stage(
                        name=name,
                        version=mv.version,
                        stage="Archived",
                    )
                    archived.append((name, mv.version))

            # Phase 2: promote all new versions
            for name in required_names:
                version = str(new_versions[name])
                self.client.transition_model_version_stage(
                    name=name,
                    version=version,
                    stage="Production",
                    archive_existing_versions=False,
                )
                promoted.append((name, version))

        except Exception as e:
            logger.exception("Ensemble promotion failed. Attempting rollback: %s", e)
            # Rollback any partial promotions: demote newly promoted back to Staging
            for name, version in reversed(promoted):
                try:
                    self.client.transition_model_version_stage(
                        name=name,
                        version=version,
                        stage="Staging",
                    )
                except Exception:  # pragma: no cover - best-effort rollback
                    logger.exception(
                        "Rollback warning: failed to revert %s v%s to Staging",
                        name,
                        version,
                    )

            # Restore archived Production versions
            for name, version in reversed(archived):
                try:
                    self.client.transition_model_version_stage(
                        name=name,
                        version=version,
                        stage="Production",
                    )
                except Exception:  # pragma: no cover - best-effort rollback
                    logger.exception(
                        "Rollback warning: failed to restore Production for %s v%s",
                        name,
                        version,
                    )

            # Re-raise to make the failure visible to the caller
            raise

        logger.info(
            "Successfully promoted challenger ensemble to Production: %s", new_versions
        )
