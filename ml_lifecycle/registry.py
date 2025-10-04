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

from typing import Dict, List, Mapping, Optional, Iterable

import logging

import mlflow
from mlflow.tracking import MlflowClient

# Provide a fallback for mlflow.register_model to enable patching in tests
# Some environments may load a minimal mlflow namespace lacking this attribute.
if not hasattr(mlflow, "register_model"):  # pragma: no cover - test-time safeguard

    def _missing_register_model(*args, **kwargs):  # pragma: no cover
        raise AttributeError("mlflow.register_model is not available")

    try:  # Avoid failing if mlflow module is immutable in this environment
        setattr(mlflow, "register_model", _missing_register_model)
    except Exception:  # pragma: no cover - best-effort only
        pass

__all__ = ["ModelRegistryManager"]

logger = logging.getLogger(__name__)


_DEFAULT_HORIZONS: tuple[str, ...] = ("1h", "4h", "8h")
_DEFAULT_MODEL_NAMES: Mapping[str, str] = {
    "1h": "uw-catboost-1h",
    "4h": "uw-catboost-4h",
    "8h": "uw-catboost-8h",
}


class ModelRegistryManager:
    """Handles registration and promotion of the CatBoost model ensemble in MLflow.

    This manager supports a configurable set of horizons and model names. By
    default, it works with ("1h","4h","8h"). For test runs, you can
    instantiate with horizons=("15m",) and model_names={"15m": "uw-catboost-15m"}.

    Methods
    -------
    register_models(parent_run_id: str) -> Dict[str, int]
        Register the child-run models under their registry names and return
        a mapping of model name to the new version number.

    promote_challenger_ensemble(new_versions: Dict[str, int]) -> None
        Promote the provided model versions to Production. If any step fails,
        the method attempts to roll back to the original state and raises.
    """

    def __init__(
        self,
        horizons: Optional[Iterable[str]] = None,
        model_names: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.client = MlflowClient()
        self.horizons = tuple(horizons) if horizons is not None else _DEFAULT_HORIZONS
        self.model_names = (
            dict(model_names) if model_names is not None else dict(_DEFAULT_MODEL_NAMES)
        )

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
                if horizon in self.horizons:
                    horizon_to_run_id[horizon] = r.info.run_id

        # Only require child runs for configured horizons
        missing = [h for h in self.horizons if h not in horizon_to_run_id]
        if missing:
            raise ValueError(
                "Missing expected child runs for horizons: " + ", ".join(missing)
            )

        # Register each child run's model artifact under its registry name
        name_to_version: Dict[str, int] = {}
        for horizon in self.horizons:
            child_run_id = horizon_to_run_id[horizon]
            model_name = self.model_names[horizon]

            # Skip horizons explicitly marked as skipped by the trainer
            try:
                child_run = self.client.get_run(child_run_id)
                skipped_flag = str(
                    child_run.data.params.get("skipped_empty_data", "")
                ).lower()
                if skipped_flag in {"true", "1", "yes"}:
                    logger.info(
                        "Skipping registration for horizon=%s (no training data; run_id=%s)",
                        horizon,
                        child_run_id,
                    )
                    continue
            except Exception:  # pragma: no cover - defensive
                pass

            # Default artifact path used by trainer
            artifact_uri = f"runs:/{child_run_id}/model"
            logger.info(
                "Registering model for horizon=%s as name=%s from artifact_uri=%s",
                horizon,
                model_name,
                artifact_uri,
            )
            try:
                mv = mlflow.register_model(model_uri=artifact_uri, name=model_name)
            except Exception as e:
                logger.warning(
                    "Registration skipped for horizon=%s (no logged model at %s): %s",
                    horizon,
                    artifact_uri,
                    e,
                )
                continue

            # mlflow.entities.model_registry.ModelVersion.version is typically str-like
            try:
                version_int = int(getattr(mv, "version"))
            except Exception:  # pragma: no cover - defensive
                version_int = int(str(getattr(mv, "version")))

            name_to_version[model_name] = version_int

        if not name_to_version:
            raise ValueError(
                "No models were registered; all horizons appear to have been skipped or lacked logged artifacts."
            )
        return name_to_version

    # -------------------- Promotion --------------------
    def promote_challenger_ensemble(self, new_versions: Dict[str, int]) -> None:
        """Promote the provided model versions by updating the 'production' alias atomically.

        This avoids deprecated stage transitions (and their known YAML serialization issues
        in certain MLflow file store versions) by using model aliases. If any step fails,
        the method attempts to roll back aliases to their previous versions and raises.

        Parameters
        ----------
        new_versions: Dict[str, int]
            Mapping from model name (e.g., "uw-catboost-1h") to new version
            numbers returned by `register_models`.
        """
        # Validate that provided versions correspond to configured model names
        # Preserve deterministic order aligned with self.horizons
        required_names = [self.model_names[h] for h in self.horizons]
        provided_names = set(new_versions.keys())
        missing = set(required_names) - provided_names
        if missing:
            raise ValueError(
                "new_versions is missing required model name(s): "
                + ", ".join(sorted(missing))
            )

        # Pre-validate that the target new versions exist
        for name in required_names:
            ver = new_versions[name]
            _ = self.client.get_model_version(name=name, version=str(ver))

        alias = "production"

        # Capture current alias mapping for rollback
        previous_alias_versions: Dict[str, Optional[str]] = {}
        for name in required_names:
            try:
                reg = self.client.get_registered_model(name)
                prev = reg.aliases.get(alias)
            except Exception:
                prev = None
            previous_alias_versions[name] = prev

        updated: List[tuple[str, str]] = []  # list of (name, version) applied

        try:
            for model_name in required_names:
                new_version_number = str(new_versions[model_name])
                logger.info(
                    "Setting alias '%s' for %s to version=%s",
                    alias,
                    model_name,
                    new_version_number,
                )
                self.client.set_registered_model_alias(
                    name=model_name, alias=alias, version=new_version_number
                )
                updated.append((model_name, new_version_number))
        except Exception as e:
            logger.exception(
                "Ensemble promotion (alias update) failed. Attempting rollback: %s", e
            )
            # Roll back aliases to previous versions
            for name, _ver in reversed(updated):
                prev = previous_alias_versions.get(name)
                try:
                    if prev is None:
                        # Remove alias if it didn't exist previously
                        self.client.delete_registered_model_alias(
                            name=name, alias=alias
                        )
                    else:
                        self.client.set_registered_model_alias(
                            name=name, alias=alias, version=str(prev)
                        )
                except Exception:  # pragma: no cover - best-effort rollback
                    logger.exception(
                        "Rollback warning: failed to restore alias '%s' for %s to %s",
                        alias,
                        name,
                        prev,
                    )
            raise

        logger.info(
            "Successfully promoted challenger ensemble by alias to Production: %s",
            new_versions,
        )
