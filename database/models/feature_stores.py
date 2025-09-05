"""Feature store ORM models with dynamically generated feature columns.

This module defines three TimescaleDB-backed wide tables for ML training
feature sets at different prediction horizons (1h, 4h, 8h). Columns are
built dynamically by parsing the feature specification JSON and adding a
mapped column for each feature flagged as `is_ml_training_feature`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
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


SPEC_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "specifications"
    / "Unicorn_Wealth_Feature_Set.json"
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


def _load_training_features(spec_path: Path = SPEC_PATH) -> List[Dict]:
    """Load and filter ML training features from JSON specification.

    Returns a list of feature dicts where `is_ml_training_feature` is True.
    Silently ignores malformed entries that lack required keys.
    """
    with spec_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    features: List[Dict] = []
    for entry in data:
        try:
            if bool(entry.get("is_ml_training_feature")):
                # Ensure required fields exist
                op = entry.get("operation")
                dtype = entry.get("output_data_type")
                if op and dtype:
                    features.append(entry)
        except Exception:
            # Skip any problematic entries
            continue
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


_TRAINING_FEATURES = _load_training_features()


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
