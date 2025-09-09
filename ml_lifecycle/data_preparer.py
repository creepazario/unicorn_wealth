"""Data preparation module for ML training.

- Loads final labeled datasets per horizon from Parquet.
- Identifies and casts categorical features using the feature spec JSON.
- Splits data chronologically into train/validation/test sets (no leakage).
- Separates features from target columns.
- Generates time-decayed sample weights for the training set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:  # Prefer relative import when available
    from ..config import (
        TIME_DECAY_HALF_LIFE_MONTHS,
        TIME_DECAY_WEIGHT_FLOOR,
        TRAIN_TEST_SPLIT_PCT,
        VALIDATION_SPLIT_PCT,
    )
except Exception:  # pragma: no cover - fallback for direct execution context
    from unicorn_wealth.config import (  # type: ignore
        TIME_DECAY_HALF_LIFE_MONTHS,
        TIME_DECAY_WEIGHT_FLOOR,
        TRAIN_TEST_SPLIT_PCT,
        VALIDATION_SPLIT_PCT,
    )


__all__ = ["PreparedData", "DataPreparer"]


@dataclass
class PreparedData:
    """Container for prepared datasets of a single forecast horizon."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    sample_weights: pd.Series


class DataPreparer:
    """Prepare data for ML training across multiple horizons.

    Parameters
    ----------
    feature_spec_path:
        Optional custom path to Unicorn_Wealth_Feature_Set.json. If not
        provided, defaults to `specifications/Unicorn_Wealth_Feature_Set.json`
        relative to the project root.
    output_dir:
        Base directory containing the Parquet datasets.
        Defaults to `output/training_data` relative to project root.
    horizons:
        Iterable of horizons to process; defaults to ("1h", "4h", "8h").
    """

    def __init__(
        self,
        feature_spec_path: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
        horizons: Optional[Iterable[str]] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.feature_spec_path = (
            Path(feature_spec_path)
            if feature_spec_path is not None
            else project_root / "specifications" / "Unicorn_Wealth_Feature_Set.json"
        )
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else project_root / "output" / "training_data"
        )
        self.horizons = tuple(horizons) if horizons is not None else ("1h", "4h", "8h")

        self._categorical_features = self._load_categorical_features()

    # --------------------------- public API ---------------------------
    def prepare_data_for_training(self) -> Dict[str, PreparedData]:
        """Load, split, type-cast, and weight data across all horizons.

        Returns
        -------
        Dict[str, PreparedData]
            Mapping from horizon (e.g., "1h") to prepared datasets and weights.
        """
        results: Dict[str, PreparedData] = {}
        for horizon in self.horizons:
            df = self._load_horizon_dataframe(horizon)
            if df.empty:
                # Return empty structures if no data is available
                empty_pd = PreparedData(
                    X_train=df.copy(),
                    y_train=pd.Series(dtype="float64"),
                    X_val=df.copy(),
                    y_val=pd.Series(dtype="float64"),
                    X_test=df.copy(),
                    y_test=pd.Series(dtype="float64"),
                    sample_weights=pd.Series(dtype="float64"),
                )
                results[horizon] = empty_pd
                continue

            # Identify and cast categorical features present in this DF
            df = self._cast_categoricals(df)

            # Chronological split
            X_train, y_train, X_val, y_val, X_test, y_test = self._time_based_split(
                df, target_col=f"target_{horizon}"
            )

            # Sample weights for training based on the timestamps of training rows
            train_timestamps = df.loc[X_train.index, "timestamp"]
            sample_weights = self._compute_time_decay_weights(train_timestamps)

            results[horizon] = PreparedData(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                sample_weights=sample_weights,
            )
        return results

    # ------------------------- internal helpers ----------------------
    def _load_categorical_features(self) -> List[str]:
        """Parse feature spec to get a list of categorical feature names.

        The column names are assumed to be the values of the "operation" field
        for entries where "is_categorical_feature" is true.
        """
        if not self.feature_spec_path.exists():
            return []

        try:
            spec = pd.read_json(self.feature_spec_path)
        except ValueError:
            # Fall back to manual JSON loading if pandas has trouble
            import json

            with open(self.feature_spec_path, "r", encoding="utf-8") as f:
                spec = pd.DataFrame(json.load(f))

        cat_mask = spec.get("is_categorical_feature", pd.Series(dtype=bool)).astype(
            bool
        )
        ops = spec.get("operation", pd.Series(dtype=str)).astype(str)
        categorical_features = list(ops[cat_mask].dropna().unique())
        return categorical_features

    def _load_horizon_dataframe(self, horizon: str) -> pd.DataFrame:
        """Load and normalize a single horizon dataframe from Parquet."""
        parquet_path = self.output_dir / f"ml_training_dataset_{horizon}.parquet"
        if not parquet_path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(parquet_path)
        df = self._ensure_timestamp_column(df)
        # Sort chronologically to avoid leakage on slicing
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    @staticmethod
    def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a 'timestamp' column exists and is datetime64[ns, UTC]."""
        out = df.copy()
        if "timestamp" not in out.columns:
            # Try to recover from index or common alternatives
            if isinstance(out.index, pd.DatetimeIndex):
                out["timestamp"] = out.index
            else:
                for cand in ("time", "date", "datetime"):
                    if cand in out.columns:
                        out["timestamp"] = out[cand]
                        break
                if "timestamp" not in out.columns:
                    out["timestamp"] = pd.to_datetime(
                        out.index, utc=True, errors="coerce"
                    )
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).copy()
        return out

    def _cast_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast known categorical columns to pandas 'category' dtype if present."""
        out = df.copy()
        for col in self._categorical_features:
            if col in out.columns:
                try:
                    out[col] = out[col].astype("category")
                except Exception:
                    # Best-effort casting: coerce to string then category
                    out[col] = out[col].astype("string").astype("category")
        return out

    def _time_based_split(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
    ]:
        """Split into train, validation, and test sets by time.

        - Validation split percentage is of the total dataset, not the training set.
        - Train comes first, followed by validation, then test (no leakage).
        """
        if target_col not in df.columns:
            # If target is missing, create a NaN series to avoid KeyError
            df = df.copy()
            df[target_col] = np.nan

        n = len(df)
        if n == 0:
            empty_df = pd.DataFrame(columns=[c for c in df.columns if c != target_col])
            empty_series = pd.Series(dtype=df[target_col].dtype)
            return (
                empty_df,
                empty_series,
                empty_df,
                empty_series,
                empty_df,
                empty_series,
            )

        # Compute sizes with safeguards
        train_pct = max(0.0, float(TRAIN_TEST_SPLIT_PCT) - float(VALIDATION_SPLIT_PCT))
        val_pct = float(VALIDATION_SPLIT_PCT)
        train_n = int(n * train_pct)
        val_n = int(n * val_pct)
        # Ensure we don't exceed n
        if train_n + val_n > n:
            overflow = train_n + val_n - n
            # Reduce validation first, then train if needed
            reduce_val = min(overflow, val_n)
            val_n -= reduce_val
            overflow -= reduce_val
            if overflow > 0:
                train_n = max(0, train_n - overflow)
        test_start = train_n + val_n
        # Slices
        train_df = df.iloc[:train_n]
        val_df = df.iloc[train_n:test_start]
        test_df = df.iloc[test_start:]

        # Separate features and targets (drop target and timestamp from X)
        feature_cols = [c for c in df.columns if c not in {target_col, "timestamp"}]
        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col].copy()
        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_col].copy()
        X_test = test_df[feature_cols].copy()
        y_test = test_df[target_col].copy()
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _compute_time_decay_weights(self, timestamps: pd.Series) -> pd.Series:
        """Compute exponential time-decay sample weights for given timestamps.

        Weights are scaled to [TIME_DECAY_WEIGHT_FLOOR, 1.0]. More recent rows
        get higher weight. The time delta is measured in months relative to the
        most recent timestamp in the provided Series.
        """
        if timestamps is None or len(timestamps) == 0:
            return pd.Series(dtype="float64")

        ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
        # Compute delta months relative to most recent timestamp
        t_max = ts.max()
        # Compute delta in days as floating point
        deltas_days = (t_max - ts).dt.total_seconds() / (60 * 60 * 24)
        # Convert days to months (average Gregorian month length)
        delta_months = deltas_days / 30.4375

        # Exponential decay using half-life
        lam = np.log(2.0) / float(TIME_DECAY_HALF_LIFE_MONTHS)
        weight_raw = np.exp(-lam * delta_months)

        # Scale to [floor, 1]
        floor = float(TIME_DECAY_WEIGHT_FLOOR)
        weights = floor + (1.0 - floor) * weight_raw

        return pd.Series(weights, index=timestamps.index)
