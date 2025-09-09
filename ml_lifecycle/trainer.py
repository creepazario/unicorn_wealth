"""Model training orchestrator for horizon-specialized CatBoost models.

This module defines the ModelTrainer class, which coordinates training for the
three horizons ("1h", "4h", "8h"). It uses GPU acceleration when available and
logs parameters, metrics, and model artifacts to MLflow using nested runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

import mlflow

try:  # optional flavor registration; do not fail tests if missing
    import mlflow.catboost  # noqa: F401  # required for model logging flavor registration
except Exception:  # pragma: no cover - optional dependency not available in CI/tests
    pass

try:  # Prefer relative import when available
    from .data_preparer import PreparedData
except Exception:  # pragma: no cover - fallback for direct execution context
    from unicorn_wealth.ml_lifecycle.data_preparer import PreparedData  # type: ignore


__all__ = ["ModelTrainer"]


def _infer_categorical_features(df: pd.DataFrame) -> List[str]:
    """Infer categorical feature columns by pandas dtype.

    CatBoost accepts column names for cat_features when passing a pandas DataFrame.
    We detect columns with dtype "category" or "object".
    """
    if df is None or df.empty:
        return []
    return [c for c in df.columns if str(df[c].dtype) in ("category", "object")]


@dataclass
class _TrainResult:
    model: CatBoostClassifier
    metrics: Mapping[str, float]
    used_task_type: str


class ModelTrainer:
    """Orchestrates training of CatBoost models for multiple horizons.

    Parameters
    ----------
    prepared_data:
        Mapping from horizon string (e.g., "1h") to PreparedData containing
        train/val/test splits and sample weights produced by DataPreparer.
    experiment_name:
        Optional MLflow experiment name to set before running. If not provided,
        the current MLflow experiment remains unchanged.
    horizons:
        Optional iterable of horizons to train. Defaults to ("1h", "4h", "8h").
    """

    def __init__(
        self,
        prepared_data: Mapping[str, PreparedData],
        experiment_name: Optional[str] = None,
        horizons: Optional[Iterable[str]] = None,
    ) -> None:
        self.prepared_data = dict(prepared_data)
        self.horizons = tuple(horizons) if horizons is not None else ("1h", "4h", "8h")
        self.experiment_name = experiment_name

    def _train_single_horizon(self, horizon: str, data: PreparedData) -> _TrainResult:
        """Train a single CatBoost model on a given horizon and compute metrics."""
        X_train, y_train = data.X_train, data.y_train
        X_val, y_val = data.X_val, data.y_val
        sample_weight = data.sample_weights

        # Drop rows with NaN targets from training and validation
        train_mask = y_train.notna()
        X_train = X_train.loc[train_mask]
        y_train = y_train.loc[train_mask]
        val_mask = y_val.notna()
        X_val_fit = X_val.loc[val_mask]
        y_val_fit = y_val.loc[val_mask]

        # Early exit if no data
        if X_train.empty or y_train.empty:
            raise ValueError(f"No training data available for horizon {horizon}")

        # Infer categorical feature names
        cat_features = _infer_categorical_features(X_train)

        # Prepare sample weights aligned to training index
        weights_arr = (
            sample_weight.reindex(X_train.index).to_numpy()
            if len(sample_weight)
            else None
        )

        # Instantiate classifier with GPU preference as required
        params: Dict[str, object] = {
            "task_type": "GPU",
            # Rely on CatBoost defaults; CatBoost auto-detects multiclass if applicable.
        }
        model = CatBoostClassifier(**params)

        # Fit with GPU, fallback to CPU if GPU is unavailable
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val_fit, y_val_fit),
                sample_weight=weights_arr,
                cat_features=cat_features,
                verbose=False,
            )
            used_task_type = "GPU"
        except Exception:  # pragma: no cover - runtime-dependent (GPU availability)
            # Retry on CPU to avoid hard failure in non-GPU environments
            cpu_model = CatBoostClassifier()
            cpu_model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                sample_weight=weights_arr,
                cat_features=cat_features,
                verbose=False,
            )
            model = cpu_model
            used_task_type = "CPU"

        # Predictions and metrics on validation set
        # Ensure y_val has no NaNs and non-empty
        y_val_clean = y_val_fit
        if y_val_clean.empty:
            metrics = {
                "val_log_loss": float("nan"),
                "val_accuracy": float("nan"),
                "val_f1_macro": float("nan"),
            }
            model.set_params(_used_task_type=used_task_type)  # type: ignore[attr-defined]
            return _TrainResult(
                model=model, metrics=metrics, used_task_type=used_task_type
            )

        proba = model.predict_proba(X_val_fit)
        preds = model.predict(X_val_fit)

        # CatBoost may return predictions as strings or numbers; normalize
        if isinstance(preds, pd.Series):
            preds_arr = preds.to_numpy()
        else:
            preds_arr = np.asarray(preds).ravel()

        # For log_loss in multiclass, provide labels to ensure correct class order
        classes = getattr(model, "classes_", None)
        if classes is None:
            # Fallback: derive from y_val
            classes = np.unique(y_val_fit)
        metrics = {
            "val_log_loss": float(log_loss(y_val_fit, proba, labels=list(classes))),
            "val_accuracy": float(accuracy_score(y_val_fit, preds_arr)),
            "val_f1_macro": float(f1_score(y_val_fit, preds_arr, average="macro")),
        }

        # Attach a training param indicating actual compute backend used
        model.set_params(_used_task_type=used_task_type)  # type: ignore[attr-defined]

        return _TrainResult(model=model, metrics=metrics, used_task_type=used_task_type)

    def train_models(self) -> Dict[str, CatBoostClassifier]:
        """Train models for configured horizons with nested MLflow runs.

        Returns
        -------
        Dict[str, CatBoostClassifier]
            Mapping from horizon to the trained CatBoostClassifier instance.
        """
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        trained: Dict[str, CatBoostClassifier] = {}
        with mlflow.start_run(run_name="UnicornWealth_Training_Run"):
            for horizon in self.horizons:
                pdata = self.prepared_data.get(horizon)
                if pdata is None:
                    # No data for this horizon
                    continue

                # If no train data, create a nested run and log skip info
                if pdata.X_train.empty or pdata.y_train.empty:
                    with mlflow.start_run(run_name=f"{horizon}_model", nested=True):
                        mlflow.log_params({"skipped_empty_data": True})
                    continue

                with mlflow.start_run(run_name=f"{horizon}_model", nested=True):
                    # Train and evaluate
                    result = self._train_single_horizon(horizon, pdata)
                    model = result.model

                    # Log hyperparameters
                    try:
                        params_dict = model.get_all_params()
                    except Exception:
                        params_dict = {}
                    # Ensure actual compute backend is logged
                    params_dict = dict(params_dict)
                    used_task_type = model.get_params().get("_used_task_type")
                    if used_task_type:
                        params_dict["used_task_type"] = used_task_type
                    if params_dict:
                        mlflow.log_params(params_dict)

                    # Log computed metrics
                    mlflow.log_metrics(dict(result.metrics))

                    # Log model artifact
                    mlflow.catboost.log_model(model, artifact_path="model")

                    trained[horizon] = model

        return trained
