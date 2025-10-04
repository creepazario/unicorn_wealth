import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Dict

import pandas as pd
import pytest

from ml_lifecycle.live_calculator import LiveFeatureCalculator


@pytest.mark.asyncio
async def test_calculator_offloads_all_horizons(mocker):
    # Mock dependencies
    mock_engine = mocker.Mock(name="UnifiedFeatureEngineeringEngine")
    mock_registry = mocker.Mock(name="DataFrameRegistry")
    mock_pool = mocker.Mock(spec=ProcessPoolExecutor)

    # Patch asyncio.get_event_loop / running loop
    # In code, calculator uses asyncio.get_running_loop(); also set get_running_loop to return the same mock loop.
    mock_loop = mocker.Mock()
    mocker.patch("asyncio.get_event_loop", return_value=mock_loop)
    mocker.patch("asyncio.get_running_loop", return_value=mock_loop)

    # Prepare futures for three horizons returning different DataFrames
    def make_future(val: int):
        fut: asyncio.Future = asyncio.Future()
        fut.set_result(pd.DataFrame({"feature": [val]}))
        return fut

    future = make_future(99)

    # Configure run_in_executor to return a single future (one offload total)
    mock_loop.run_in_executor.return_value = future

    # Mock async registry methods
    async def _aget_df(name: str):
        return pd.DataFrame(
            {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}
        )

    async def _astore_df(name: str, df: pd.DataFrame, mode: str, period: int):
        return None

    mock_registry.get_df = _aget_df  # type: ignore
    mock_registry.store_df = _astore_df  # type: ignore

    # Instantiate calculator
    calc = LiveFeatureCalculator(
        engine=mock_engine, registry=mock_registry, process_pool=mock_pool
    )

    # Execute
    tokens = ["BTC"]
    result: Dict[str, pd.DataFrame] = await calc.calculate_all_horizons(tokens)

    # Assertions on executor offloading (should be exactly one offload now)
    assert mock_loop.run_in_executor.call_count == 1

    # Extract called function and args from the single run_in_executor call
    args, kwargs = mock_loop.run_in_executor.call_args
    assert len(args) >= 4
    func = args[1]
    called_token = args[2]

    # engine.run_15m_pipeline should be the target function
    assert func is mock_engine.run_15m_pipeline
    assert called_token == "BTC"

    # Assert mapping contains three resolved DataFrames
    assert set(result.keys()) == {"1h", "4h", "8h"}
    assert isinstance(result["1h"], pd.DataFrame)
    assert isinstance(result["4h"], pd.DataFrame)
    assert isinstance(result["8h"], pd.DataFrame)
    # The subsets come from the same returned wide DF (future)
    pd.testing.assert_frame_equal(
        result["1h"], future.result().loc[:, result["1h"].columns]
    )
    pd.testing.assert_frame_equal(
        result["4h"], future.result().loc[:, result["4h"].columns]
    )
    pd.testing.assert_frame_equal(
        result["8h"], future.result().loc[:, result["8h"].columns]
    )
