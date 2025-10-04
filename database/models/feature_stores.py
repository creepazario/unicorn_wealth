"""Feature store ORM models with statically defined feature columns.

This module defines three TimescaleDB-backed wide tables for ML training
feature sets at different prediction horizons (1h, 4h, 8h).

At import time we derive the list of training features from the
specifications, including any configured lagged features, and then define
static columns accordingly. This keeps runtime operations decoupled from
spec parsing while ensuring schema matches the blueprint.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Optional, Set

import json
from pathlib import Path

from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy.types import BigInteger, Boolean, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base

# TimescaleDB hypertable configuration identical to raw_data models
TIMESCALEDB_ARGS: Tuple[Dict[str, Dict[str, str]], ...] = (
    {"timescaledb_hypertable": {"time_column_name": "timestamp"}},
)


def _dtype_to_sa(dtype: str):
    """Map feature output_data_type string to SQLAlchemy column type.

    Defaults to Float when the dtype is unknown.
    """
    normalized = (dtype or "").strip().lower()
    mapping = {
        "float64": Float,
        "float32": Float,
        "double": Float,
        "float": Float,
        "int64": BigInteger,
        "int32": Integer,
        "int": Integer,
        "boolean": Boolean,
        "bool": Boolean,
        "string": String,
        "str": String,
    }
    return mapping.get(normalized, Float)


def _collect_training_features_from_specs() -> List[Dict]:
    """Build training feature definitions from JSON specifications.

    - Reads lowercase 'unicorn_wealth_feature_set.json' to include all
      operations with is_ml_training_feature=true and generate lag features
      for flags ml_feature_lag_1/2/4.
    - Reads capitalized 'Unicorn_Wealth_Feature_Set.json' to ensure legacy
      base operations used by tests are present.
    - Deduplicates and validates names, defaulting dtype to float64 when
      missing; forces 'economic_event' to boolean if dtype absent.
    """
    base_dir = Path(__file__).resolve().parents[2]  # project root
    spec_dir = base_dir / "specifications"
    lower_path = spec_dir / "unicorn_wealth_feature_set.json"
    upper_path = spec_dir / "Unicorn_Wealth_Feature_Set.json"

    features: List[Dict] = []
    seen: Set[str] = set()
    reserved: Set[str] = {"timestamp", "token"}

    def _add(name: Optional[str], dtype: Optional[str]) -> None:
        if not name:
            return
        name_s = str(name).strip()
        if not name_s or name_s in seen or name_s in reserved:
            return
        if not name_s.isidentifier():
            return
        seen.add(name_s)
        features.append({"operation": name_s, "output_data_type": dtype or "float64"})

    # Load comprehensive spec with lag definitions
    try:
        if lower_path.exists():
            with lower_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data or []:
                if not isinstance(item, dict):
                    continue
                if not item.get("is_ml_training_feature"):
                    continue
                op = item.get("operation")
                dtype = item.get("output_data_type") or (
                    "boolean" if op == "economic_event" else "float64"
                )
                _add(op, dtype)
                # Add lag features as requested by issue
                for key, n in (
                    ("ml_feature_lag_1", 1),
                    ("ml_feature_lag_2", 2),
                    ("ml_feature_lag_4", 4),
                ):
                    if item.get(key) is True and isinstance(op, str) and op:
                        _add(f"{op}_lag_{n}", dtype)
    except Exception:
        # Ignore spec loading errors; fallback will handle
        pass

    # Ensure base operations from the smaller legacy spec are included
    try:
        if upper_path.exists():
            with upper_path.open("r", encoding="utf-8") as f:
                data_u = json.load(f)
            for item in data_u or []:
                if not isinstance(item, dict):
                    continue
                if not item.get("is_ml_training_feature"):
                    continue
                op = item.get("operation")
                dtype = "boolean" if op == "economic_event" else "float64"
                _add(op, dtype)
    except Exception:
        pass

    return features


def _static_training_features() -> List[Dict]:
    """Return static training feature definitions.

    The list below must mirror the final training set produced by the
    UnifiedFeatureEngine historical pipeline (is_ml_training_feature=true).
    We include ADX components for 15m/1h/4h/1d, RSI 15m, atr_normalized for
    15m/1h/4h/1d, temporal cyclical features, session/kill-zone features,
    and the external boolean economic_event flag.
    All numeric are stored as float64 for consistency; economic_event is boolean.
    """
    cols = [
        "rsi_15m",
        "atr_normalized_15m",
        "atr_normalized_1h",
        "atr_normalized_4h",
        "atr_normalized_1d",
        "adx_15m",
        "adx_pos_15m",
        "adx_neg_15m",
        "adx_1h",
        "adx_pos_1h",
        "adx_neg_1h",
        "adx_4h",
        "adx_pos_4h",
        "adx_neg_4h",
        "adx_1d",
        "adx_pos_1d",
        "adx_neg_1d",
        # Temporal cyclical time features (UTC)
        "day_of_week_cos",
        "day_of_week_sin",
        "hour_of_day_cos",
        "hour_of_day_sin",
        "minute_of_hour_cos",
        "minute_of_hour_sin",
        # Sessions & Kill-zone features (15m base)
        "sess_ny_flag",
        "sess_london_flag",
        "sess_asia_flag",
        "kz_ny_flag",
        "kz_london_flag",
        "friday_flag",
        "bars_to_session_close",
        "bars_to_kz_end",
        "bars_since_session_open",
        "bars_since_midnight_utc",
        "bars_since_kz_start",
    ]
    features = [{"operation": c, "output_data_type": "float64"} for c in cols]
    # Add external boolean feature per spec (unicorn_wealth_feature_set.json operation: economic_event)
    features.append({"operation": "economic_event", "output_data_type": "boolean"})
    return features


def _add_feature_columns(cls, features: Iterable[Dict]) -> None:
    """Dynamically add mapped columns to a SQLAlchemy declarative class.

    For each feature, adds a column named by `operation` with a type mapped
    from `output_data_type`.
    """
    seen: set[str] = set()
    reserved = {"timestamp", "token"}
    for feat in features:
        name = str(feat["operation"]).strip()
        if not name or name in seen or name in reserved:
            continue
        if not name.isidentifier():
            # Skip any invalid Python identifiers to avoid attribute errors
            continue
        seen.add(name)
        col_type = _dtype_to_sa(str(feat.get("output_data_type", "")))
        # Attach the column to the class
        setattr(cls, name, mapped_column(col_type))


# Build from specs; fallback to static list if anything goes wrong or empty
try:
    _TRAINING_FEATURES = _collect_training_features_from_specs()
    if not _TRAINING_FEATURES:
        _TRAINING_FEATURES = _static_training_features()
except Exception:
    _TRAINING_FEATURES = _static_training_features()


class FeatureStore1h(Base):
    __tablename__ = "feature_store_1h"
    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "token"),
        *TIMESCALEDB_ARGS,
    )

    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)
    token: Mapped[str] = mapped_column(String, nullable=False)


# Dynamically add all training feature columns
_add_feature_columns(FeatureStore1h, _TRAINING_FEATURES)


class FeatureStore4h(Base):
    __tablename__ = "feature_store_4h"
    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "token"),
        *TIMESCALEDB_ARGS,
    )

    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)
    token: Mapped[str] = mapped_column(String, nullable=False)


_add_feature_columns(FeatureStore4h, _TRAINING_FEATURES)


class FeatureStore8h(Base):
    __tablename__ = "feature_store_8h"
    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "token"),
        *TIMESCALEDB_ARGS,
    )

    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)
    token: Mapped[str] = mapped_column(String, nullable=False)


_add_feature_columns(FeatureStore8h, _TRAINING_FEATURES)


__all__ = [
    "FeatureStore1h",
    "FeatureStore4h",
    "FeatureStore8h",
]
