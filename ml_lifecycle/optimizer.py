from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

try:  # Optional at import time for static analyzers
    import optuna
except Exception:  # pragma: no cover
    optuna = None  # type: ignore[assignment]

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]

from .data_preparer import DataPreparer, PreparedData
from .optimization_space import (
    get_feature_optimization_space,
    get_model_optimization_space,
)
from .trainer import ModelTrainer
from .validator import ModelValidator


@dataclass
class WFOWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


class StrategyOptimizer:
    """Run a two-stage Walk-Forward Optimization (WFO) with Optuna and MLflow.

    This orchestrates DataPreparer, ModelTrainer, and ModelValidator across
    rolling time windows. It supports two stages: 'features' and 'models'.

    Parameters
    ----------
    dataset : pd.DataFrame
        Master historical dataset containing features, targets, prices, and
        a 'timestamp' column. Expected columns include:
        - timestamp (datetime-like)
        - OHLCV columns for price backtesting: [open, high, low, close]
        - feature columns used by models
        - target columns per horizon: target_1h, target_4h, target_8h
    config_module : module
        Module providing WFO_SETTINGS with keys TRAINING_WINDOW_MONTHS,
        VALIDATION_WINDOW_MONTHS, STEP_SIZE_MONTHS.
    horizons : Optional[Iterable[str]]
        Horizons to optimize. Defaults to ("1h", "4h", "8h").
    experiment_name : Optional[str]
        MLflow experiment name to log runs under.
    n_trials : int
        Number of Optuna trials to run. Default 20.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        config_module: Any,
        horizons: Optional[Iterable[str]] = None,
        experiment_name: Optional[str] = None,
        n_trials: int = 20,
    ) -> None:
        self.master_df = dataset.copy()
        self.master_df["timestamp"] = pd.to_datetime(
            self.master_df["timestamp"], utc=True, errors="coerce"
        )
        self.master_df = self.master_df.dropna(subset=["timestamp"]).sort_values(
            "timestamp"
        )
        self.config = config_module
        self.horizons = tuple(horizons) if horizons is not None else ("1h", "4h", "8h")
        self.experiment_name = experiment_name or "UnicornWealth_WFO"
        self.n_trials = int(n_trials)
        self._stage: str = "features"
        self._wfo_windows: List[WFOWindow] = []
        # Precompute windows once
        self._wfo_windows = self._generate_wfo_windows()

    # ------------------------------ WFO windows ------------------------------
    def _generate_wfo_windows(self) -> List[WFOWindow]:
        """Generate rolling train/validation windows based on config settings.

        Returns a list of WFOWindow with inclusive [start, end) semantics for slicing.
        """
        settings: Mapping[str, Any] = getattr(self.config, "WFO_SETTINGS", {})
        train_m = int(settings.get("TRAINING_WINDOW_MONTHS", 12))
        val_m = int(settings.get("VALIDATION_WINDOW_MONTHS", 3))
        step_m = int(settings.get("STEP_SIZE_MONTHS", 3))

        if self.master_df.empty:
            return []

        t_min = pd.to_datetime(self.master_df["timestamp"].min(), utc=True)
        t_max = pd.to_datetime(self.master_df["timestamp"].max(), utc=True)

        windows: List[WFOWindow] = []
        cur_start = t_min
        while True:
            train_end = cur_start + pd.DateOffset(months=train_m)
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=val_m)
            # Stop if validation end exceeds data range
            if val_end > t_max:
                break
            windows.append(
                WFOWindow(
                    train_start=cur_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                )
            )
            # Advance by step size
            cur_start = cur_start + pd.DateOffset(months=step_m)
            # Safety to avoid infinite loop in degenerate cases
            if cur_start >= t_max:
                break
        return windows

    # ------------------------------ Optuna objective ------------------------------
    def _objective(self, trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
        """Optuna objective that averages validation Sharpe across WFO windows."""
        if optuna is None:  # pragma: no cover
            raise RuntimeError("optuna is required to run optimization.")

        self._ensure_mlflow_experiment()

        # Sample hyperparameters for this stage
        if self._stage == "features":
            params = get_feature_optimization_space(trial)
        elif self._stage == "models":
            params = get_model_optimization_space(trial)
        else:
            raise ValueError("stage must be 'features' or 'models'")

        # Start a nested run for this trial
        if mlflow is not None:
            mlflow.log_params({"stage": self._stage})
            mlflow.log_params({f"param__{k}": v for k, v in params.items()})

        sharpe_scores: List[float] = []
        # Loop over WFO windows
        for i, w in enumerate(self._wfo_windows):
            # Nested run for window
            if mlflow is not None:
                mlflow.start_run(run_name=f"window_{i}", nested=True)
                mlflow.log_params(
                    {
                        "train_start": str(w.train_start),
                        "train_end": str(w.train_end),
                        "val_start": str(w.val_start),
                        "val_end": str(w.val_end),
                    }
                )

            # Slice data
            train_mask = (self.master_df["timestamp"] >= w.train_start) & (
                self.master_df["timestamp"] < w.train_end
            )
            val_mask = (self.master_df["timestamp"] >= w.val_start) & (
                self.master_df["timestamp"] < w.val_end
            )
            train_df = self.master_df.loc[train_mask].copy()
            val_df = self.master_df.loc[val_mask].copy()

            # Prepare data per horizon using DataPreparer helpers
            dp = DataPreparer()
            prepared: Dict[str, PreparedData] = {}
            # Build a feature set that excludes timestamp, prices, and all targets
            price_cols = [
                c for c in ["open", "high", "low", "close"] if c in train_df.columns
            ]

            def _feature_cols(df: pd.DataFrame) -> List[str]:
                return [
                    c
                    for c in df.columns
                    if c not in {"timestamp", *price_cols}
                    and not c.startswith("target_")
                ]

            for h in self.horizons:
                target_col = f"target_{h}"
                if (
                    target_col not in train_df.columns
                    or target_col not in val_df.columns
                ):
                    continue
                # Build X/y
                X_train = train_df[_feature_cols(train_df)].copy()
                y_train = train_df[target_col].copy()
                X_val = val_df[_feature_cols(val_df)].copy()
                y_val = val_df[target_col].copy()
                # No explicit test in WFO; provide empty
                X_test = X_val.iloc[0:0].copy()
                y_test = y_val.iloc[0:0].copy()
                # Compute time-decay sample weights using DataPreparer utility
                sample_weights = dp._compute_time_decay_weights(  # type: ignore[attr-defined]
                    train_df["timestamp"]
                )
                prepared[h] = PreparedData(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    sample_weights=sample_weights,
                )

            # Train models for available horizons
            if not prepared:
                # No data for any horizon; log zero and continue
                sharpe_scores.append(0.0)
                if mlflow is not None:
                    mlflow.log_metrics({"sharpe": 0.0})
                    mlflow.end_run()
                continue

            model_params: Optional[Mapping[str, object]] = (
                params if self._stage == "models" else None
            )
            trainer = ModelTrainer(
                prepared_data=prepared,
                experiment_name=self.experiment_name,
                horizons=self.horizons,
                model_params=model_params,
            )
            models = trainer.train_models()

            # Backtest on validation slice using in-memory models
            # Features for validator should match the columns the models expect
            # We will use the union of validation feature sets across horizons
            if len(prepared) == 0:
                sharpe = 0.0
            else:
                # Use features from the first available horizon as base
                any_h = next(iter(prepared.keys()))
                val_features = prepared[any_h].X_val.copy()
                # Prices must exist; if missing, score is zero
                if not price_cols or any(c not in val_df.columns for c in price_cols):
                    sharpe = 0.0
                else:
                    val_prices = val_df[price_cols].copy()
                    validator = ModelValidator(
                        test_features=val_features,
                        test_prices=val_prices,
                        challenger_run_id="",  # not used when passing models directly
                    )
                    metrics = validator._run_backtest(models, val_features, val_prices)
                    sharpe = float(metrics.sharpe_ratio)

            sharpe_scores.append(sharpe)
            if mlflow is not None:
                mlflow.log_metrics({"sharpe": sharpe})
                mlflow.end_run()

        # Average across windows
        avg_sharpe = (
            float(sum(sharpe_scores) / len(sharpe_scores)) if sharpe_scores else 0.0
        )
        # Let Optuna know this is our objective value
        return avg_sharpe

    # ------------------------------ Public API ------------------------------
    def run_optimization(self, stage: str) -> Dict[str, Any]:
        """Run the WFO study for the given stage ('features' or 'models').

        Returns the best trial params and logs them as an MLflow artifact.
        """
        if optuna is None:  # pragma: no cover
            raise RuntimeError("optuna is required to run optimization.")

        if stage not in {"features", "models"}:
            raise ValueError("stage must be 'features' or 'models'")
        self._stage = stage

        self._ensure_mlflow_experiment()
        best_params: Dict[str, Any] = {}

        # Parent run for the whole study
        if mlflow is not None:
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=f"WFO_{stage}_study")

        try:
            # Create Optuna study (maximize objective)
            study = optuna.create_study(direction="maximize")

            # Wrap objective to ensure nested trial runs
            def _wrapped_objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
                if mlflow is not None:
                    mlflow.start_run(
                        run_name=f"optuna_trial_{trial.number}", nested=True
                    )
                try:
                    score = self._objective(trial)
                    if mlflow is not None:
                        mlflow.log_metric("avg_sharpe", score)
                finally:
                    if mlflow is not None:
                        mlflow.end_run()
                return score

            study.optimize(
                _wrapped_objective, n_trials=self.n_trials, show_progress_bar=False
            )

            # Persist best params
            best_params = dict(study.best_trial.params)
            out_name = (
                "best_feature_params.json"
                if stage == "features"
                else "best_model_params.json"
            )
            out_path = Path(out_name)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2, default=str)

            # Log artifact
            if mlflow is not None:
                mlflow.log_artifact(str(out_path))
        finally:
            if mlflow is not None and getattr(mlflow, "active_run", None) is not None:
                try:
                    mlflow.end_run()
                except Exception:
                    pass

        return best_params

    # ------------------------------ Utilities ------------------------------
    def _ensure_mlflow_experiment(self) -> None:
        if mlflow is not None:
            try:
                mlflow.set_experiment(self.experiment_name)
            except Exception:
                # Ignore if MLflow not configured
                pass
