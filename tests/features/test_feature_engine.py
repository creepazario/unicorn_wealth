import json
from pathlib import Path
from typing import Any

import pytest

from unicorn_wealth.features.feature_engine import UnifiedFeatureEngine


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


def test_dag_construction(mock_feature_spec: Path) -> None:
    engine = _init_engine_with_spec(mock_feature_spec)

    # Rebuild DAG explicitly to get a handle to the returned graph
    g = engine._build_dag(engine._features)

    # Nodes should include operations and parsed sources (ending with _df)
    nodes = set(g.nodes())
    expected_ops = {"A", "B", "C", "D", "E", "F", "G"}
    expected_sources = {"A_df", "B_df", "src1_df", "src2_df"}
    assert expected_ops.issubset(nodes)
    assert expected_sources.issubset(nodes)

    # Validate edges from sources to operations via internal adjacency
    # This is a white-box check tailored for the minimal graph adapter
    adj = getattr(g, "_adj")
    assert "B" in adj.get("A_df", set())  # A_df -> B
    assert "C" in adj.get("B_df", set())  # B_df -> C
    assert "D" in adj.get("src1_df", set())  # src1_df -> D
    assert "E" in adj.get("src1_df", set())  # src1_df -> E
    assert "D" in adj.get("src2_df", set())  # src2_df -> D


def test_topological_sort_order(mock_feature_spec: Path) -> None:
    engine = _init_engine_with_spec(mock_feature_spec)

    ordered = engine._topological_sort_features()

    # Expect order primarily by step_order, stable by op name for ties
    # A(1), B(2), C(3), E(4), D(5), F(10), G(10) -> F before G alphabetically
    assert ordered.index("A") < ordered.index("B") < ordered.index("C")
    assert ordered.index("C") < ordered.index("E") < ordered.index("D")
    # Tie-break at step_order 10 should sort by name
    f_pos = ordered.index("F")
    g_pos = ordered.index("G")
    assert f_pos < g_pos
