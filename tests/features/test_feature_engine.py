import json
from pathlib import Path
from typing import Any

import pytest

from features.feature_engine import UnifiedFeatureEngine


class DummyRegistry:
    async def get_df(self, name: str) -> Any:  # pragma: no cover - not used here
        raise KeyError(name)


class DummySql:
    pass


@pytest.fixture()
def mock_feature_spec(tmp_path: Path) -> Path:
    # Build a small mock feature set
    spec = [
        {
            "operation": "A",
            "step_order": 1,
            "calculate_per_token": True,
            "transform_data_source": "",
        },
        {
            "operation": "B",
            "step_order": 2,
            "calculate_per_token": True,
            "transform_data_source": "A_df",
        },
        {
            "operation": "C",
            "step_order": 3,
            "calculate_per_token": True,
            "transform_data_source": "B_df",
        },
        {
            "operation": "D",
            "step_order": 5,
            "calculate_per_token": True,
            "transform_data_source": "src1_df src2_df",
        },
        {
            "operation": "E",
            "step_order": 4,
            "calculate_per_token": True,
            "transform_data_source": "src1_df",
        },
        {
            "operation": "F",
            "step_order": 10,
            "calculate_per_token": True,
            "transform_data_source": "",
        },
        {
            "operation": "G",
            "step_order": 10,
            "calculate_per_token": True,
            "transform_data_source": "",
        },
    ]
    path = tmp_path / "mock_feature_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _init_engine_with_spec(spec_path: Path) -> UnifiedFeatureEngine:
    registry = DummyRegistry()
    sql = DummySql()
    return UnifiedFeatureEngine(
        registry=registry,
        sql_engine=sql,  # type: ignore[arg-type]
        feature_spec_path=str(spec_path),
    )
