import sys
import types
from types import SimpleNamespace
from typing import Dict

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
from ml_lifecycle.registry import ModelRegistryManager  # noqa: E402


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
    mocker.patch("ml_lifecycle.registry.MlflowClient", return_value=client_mock)
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
        "ml_lifecycle.registry.mlflow.register_model",
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

    # No existing alias mapping
    mock_mlflow_client.get_registered_model.return_value = SimpleNamespace(aliases={})

    mgr = ModelRegistryManager()

    new_versions = {
        "uw-catboost-1h": 10,
        "uw-catboost-4h": 20,
        "uw-catboost-8h": 30,
    }

    # Act
    mgr.promote_challenger_ensemble(new_versions)

    # Assert: expect alias set operations for each model
    calls = mock_mlflow_client.set_registered_model_alias.call_args_list
    assert len(calls) == 3

    expected_seq = [
        mocker.call(
            name="uw-catboost-1h",
            alias="production",
            version=str(new_versions["uw-catboost-1h"]),
        ),
        mocker.call(
            name="uw-catboost-4h",
            alias="production",
            version=str(new_versions["uw-catboost-4h"]),
        ),
        mocker.call(
            name="uw-catboost-8h",
            alias="production",
            version=str(new_versions["uw-catboost-8h"]),
        ),
    ]

    assert calls == expected_seq
