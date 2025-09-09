import sys
import types
from types import SimpleNamespace
from typing import Dict, List

import pytest

# Stub mlflow modules before importing the module under test to avoid hard dependency
mlflow_stub = types.ModuleType("mlflow")


def _stub_register_model(
    *args, **kwargs
):  # pragma: no cover - should be patched in tests
    raise AssertionError(
        "mlflow.register_model stub called unexpectedly before patching"
    )


mlflow_stub.register_model = _stub_register_model

mlflow_tracking_stub = types.ModuleType("mlflow.tracking")


class _StubClient:  # minimal placeholder; will be replaced by mock via patch
    pass


mlflow_tracking_stub.MlflowClient = _StubClient

sys.modules.setdefault("mlflow", mlflow_stub)
sys.modules.setdefault("mlflow.tracking", mlflow_tracking_stub)

# We will import the manager under test
from unicorn_wealth.ml_lifecycle.registry import ModelRegistryManager  # noqa: E402


def _make_run(run_id: str, run_name: str):
    """Create a minimal run-like object with .info.run_id and .data.tags."""
    info = SimpleNamespace(run_id=run_id)
    data = SimpleNamespace(tags={"mlflow.runName": run_name})
    return SimpleNamespace(info=info, data=data)


def _make_model_version(version: str):
    """Create a minimal model-version-like object with .version attribute."""
    return SimpleNamespace(version=version)


@pytest.fixture
def mock_mlflow_client(mocker):
    """Patch MlflowClient in the registry module so instantiation uses a mock."""
    client_mock = mocker.MagicMock()
    mocker.patch(
        "unicorn_wealth.ml_lifecycle.registry.MlflowClient", return_value=client_mock
    )
    return client_mock


def test_register_models(mocker, mock_mlflow_client):
    # Arrange: mock parent get_run to provide experiment id
    mock_mlflow_client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(experiment_id="exp-123")
    )

    # Arrange: mock search_runs to return three child runs
    child_runs = [
        _make_run("run-1h", "1h_model"),
        _make_run("run-4h", "4h_model"),
        _make_run("run-8h", "8h_model"),
    ]
    mock_mlflow_client.search_runs.return_value = child_runs

    # Arrange: mock top-level mlflow.register_model to return versions
    register_model_mock = mocker.patch(
        "unicorn_wealth.ml_lifecycle.registry.mlflow.register_model",
        side_effect=[
            _make_model_version("1"),
            _make_model_version("2"),
            _make_model_version("3"),
        ],
    )

    mgr = ModelRegistryManager()

    # Act
    versions: Dict[str, int] = mgr.register_models(parent_run_id="parent-xyz")

    # Assert: register_model called exactly three times with correct args
    assert register_model_mock.call_count == 3
    expected_calls = [
        mocker.call(model_uri="runs:/run-1h/model", name="uw-catboost-1h"),
        mocker.call(model_uri="runs:/run-4h/model", name="uw-catboost-4h"),
        mocker.call(model_uri="runs:/run-8h/model", name="uw-catboost-8h"),
    ]
    register_model_mock.assert_has_calls(expected_calls, any_order=False)

    # Assert: returned mapping contains correct names and int versions
    assert versions == {
        "uw-catboost-1h": 1,
        "uw-catboost-4h": 2,
        "uw-catboost-8h": 3,
    }


def test_promote_challenger_ensemble(mocker, mock_mlflow_client):
    # Arrange
    # Ensure get_model_version returns something (existence check)
    mock_mlflow_client.get_model_version.side_effect = [
        _make_model_version("10"),
        _make_model_version("20"),
        _make_model_version("30"),
    ]

    # Fake existing production versions
    prod_versions = [
        _make_model_version("100"),
    ]

    def _get_latest_versions(name: str, stages: List[str]):  # noqa: ARG001
        return prod_versions

    mock_mlflow_client.get_latest_versions.side_effect = _get_latest_versions

    mgr = ModelRegistryManager()

    new_versions = {
        "uw-catboost-1h": 10,
        "uw-catboost-4h": 20,
        "uw-catboost-8h": 30,
    }

    # Act
    mgr.promote_challenger_ensemble(new_versions)

    # Assert: we expect 6 transitions (3 archive + 3 promote)
    calls = mock_mlflow_client.transition_model_version_stage.call_args_list
    assert len(calls) == 6

    # Validate that for each model we first archived existing, then promoted new
    # Since the manager processes names in order 1h, 4h, 8h, check sequence
    expected_seq = [
        # archive current production
        mocker.call(name="uw-catboost-1h", version="100", stage="Archived"),
        mocker.call(name="uw-catboost-4h", version="100", stage="Archived"),
        mocker.call(name="uw-catboost-8h", version="100", stage="Archived"),
        # promote new versions
        mocker.call(
            name="uw-catboost-1h",
            version=str(new_versions["uw-catboost-1h"]),
            stage="Production",
            archive_existing_versions=False,
        ),
        mocker.call(
            name="uw-catboost-4h",
            version=str(new_versions["uw-catboost-4h"]),
            stage="Production",
            archive_existing_versions=False,
        ),
        mocker.call(
            name="uw-catboost-8h",
            version=str(new_versions["uw-catboost-8h"]),
            stage="Production",
            archive_existing_versions=False,
        ),
    ]

    # The exact order should match implementation; assert sequence equality
    assert calls == expected_seq
