from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml_lifecycle.inference import InferenceEngine


@pytest.mark.asyncio
async def test_load_models_from_registry(mocker: Any) -> None:
    # Arrange: mock mlflow.pyfunc.load_model to avoid real MLflow access
    load_model_mock = mocker.patch(
        "mlflow.pyfunc.load_model", autospec=True, return_value=object()
    )

    # Use a small executor; load_models uses run_in_executor internally
    with ProcessPoolExecutor(max_workers=1) as executor:
        engine = InferenceEngine(executor)

        # Act
        await engine.load_models()

    # Assert: called three times with Production URIs
    expected_uris = {
        "models:/uw-catboost-1h/Production",
        "models:/uw-catboost-4h/Production",
        "models:/uw-catboost-8h/Production",
    }
    assert load_model_mock.call_count == 3
    called_uris = {call.args[0] for call in load_model_mock.call_args_list}
    assert called_uris == expected_uris


@pytest.mark.asyncio
async def test_predict_offloads_to_executor(mocker: Any) -> None:
    # Arrange: create a mock model with predict_proba
    class MockModel:
        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover
            return np.array([[0.1, 0.9]])

    mock_model = MockModel()

    # Patch asyncio.get_running_loop to return a mock loop
    loop = asyncio.get_event_loop()  # noqa: F841

    # A helper to return an already-resolved future
    async def resolved(value: Any) -> Any:
        return value

    # Configure run_in_executor to return a resolved Future with sample output
    sample_result = np.array([[0.2, 0.8]])

    async def fake_run_in_executor(executor: Any, func: Any, *args: Any) -> Any:
        # Return a predetermined result; assertions will be made after the call
        return sample_result

    loop_mock = mocker.MagicMock()
    loop_mock.run_in_executor.side_effect = fake_run_in_executor

    get_loop_patch = mocker.patch(
        "asyncio.get_running_loop", autospec=True, return_value=loop_mock
    )

    # Instantiate engine and inject mock models directly (bypass load_models)
    with ProcessPoolExecutor(max_workers=1) as executor:
        engine = InferenceEngine(executor)
        engine._model_1h = mock_model  # type: ignore[attr-defined]
        engine._model_4h = mock_model  # type: ignore[attr-defined]
        engine._model_8h = mock_model  # type: ignore[attr-defined]

        # Prepare single-row feature data for each horizon
        row = {"feat_a": 1.0, "feat_b": "x"}
        features = {
            "1h": pd.DataFrame([row]),
            "4h": pd.DataFrame([row]),
            "8h": pd.DataFrame([row]),
        }

        # Act
        result = await engine.predict(features)

    # Assert: run_in_executor called exactly three times and with predict_proba
    assert loop_mock.run_in_executor.call_count == 3
    for call in loop_mock.run_in_executor.call_args_list:
        # args: (executor, func, df) as used by engine
        _, func, *_ = call.args
        # Verify it's the model's predict_proba bound method
        assert getattr(func, "__name__", None) == "predict_proba"
        assert getattr(func, "__self__", None) is mock_model

    # Also verify result structure has entries for each horizon
    assert set(result.keys()) == {"1h", "4h", "8h"}

    # Ensure our patch was used
    assert get_loop_patch.called
