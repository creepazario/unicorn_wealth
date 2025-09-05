import sys
import json
from pathlib import Path

# Ensure project root is on sys.path for direct package imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.models.feature_stores import (  # noqa: E402
    FeatureStore1h,
    FeatureStore4h,
    FeatureStore8h,
)
from sqlalchemy import PrimaryKeyConstraint  # noqa: E402


SPEC_PATH = (
    Path(__file__).resolve().parents[3]
    / "specifications"
    / "Unicorn_Wealth_Feature_Set.json"
)

RESERVED = {"timestamp", "token"}


def _expected_feature_names():
    with SPEC_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    names = []
    for item in data:
        try:
            if item.get("is_ml_training_feature"):
                op = item.get("operation")
                if op and isinstance(op, str):
                    # Only keep valid Python identifiers
                    # The model creation step skips invalid names
                    if op.isidentifier() and op not in RESERVED:
                        names.append(op)
        except Exception:
            continue
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def _assert_model(model, expected_tablename: str):
    # __tablename__ check
    assert getattr(model, "__tablename__", None) == expected_tablename

    # Dynamic columns check
    for col in _expected_feature_names():
        assert hasattr(model, col), f"Missing feature column on {model.__name__}: {col}"

    # __table_args__ should include a PrimaryKeyConstraint
    # and the TimescaleDB hypertable config
    table_args = getattr(model, "__table_args__", ()) or ()

    # Normalize to tuple
    if isinstance(table_args, dict):
        table_args = (table_args,)  # pragma: no cover - not expected here

    has_pk = any(isinstance(arg, PrimaryKeyConstraint) for arg in table_args)

    def _is_ts(arg):
        if not isinstance(arg, dict):
            return False
        cfg = arg.get("timescaledb_hypertable")
        if not isinstance(cfg, dict):
            return False
        return cfg.get("time_column_name") == "timestamp"

    has_timescale = any(_is_ts(arg) for arg in table_args)

    assert (
        has_pk
    ), "__table_args__ must contain a PrimaryKeyConstraint('timestamp','token')"
    assert (
        has_timescale
    ), "__table_args__ must include TimescaleDB hypertable configuration"


def test_feature_store_1h_model():
    _assert_model(FeatureStore1h, "feature_store_1h")


def test_feature_store_4h_model():
    _assert_model(FeatureStore4h, "feature_store_4h")


def test_feature_store_8h_model():
    _assert_model(FeatureStore8h, "feature_store_8h")
