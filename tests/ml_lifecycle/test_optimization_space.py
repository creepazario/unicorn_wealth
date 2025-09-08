from typing import Any, Dict, List, Tuple

import pytest

from ml_lifecycle.optimization_space import (
    get_feature_optimization_space,
    get_model_optimization_space,
)


@pytest.mark.usefixtures("mocker")
def test_get_feature_optimization_space(mocker: pytest.MonkeyPatch) -> None:
    # Arrange: create a mock trial whose suggest_* methods record calls and return stable values
    call_log_int: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    call_log_float: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

    # Deterministic return map for certain keys to make constraint checks easier
    int_returns = {
        # EMA 15m sequence
        "ema_15m__fast": 10,
        "ema_15m__mid": 20,
        "ema_15m__slow": 30,
        # MACD 15m
        "macd_15m__window_fast": 8,
        "macd_15m__window_slow": 26,
        "macd_15m__window_sign": 9,
    }

    def suggest_int_side_effect(name: str, low: int, high: int, step: int = 1) -> int:
        call_log_int.append(((name, low, high), {"step": step}))
        return int_returns.get(name, low)

    def suggest_float_side_effect(
        name: str, low: float, high: float, **kwargs: Any
    ) -> float:
        call_log_float.append(((name, low, high), kwargs))
        # Return the midpoint for determinism
        return float((low + high) / 2.0)

    trial = mocker.Mock()
    trial.suggest_int.side_effect = suggest_int_side_effect
    trial.suggest_float.side_effect = suggest_float_side_effect

    # Act
    result = get_feature_optimization_space(trial)

    # Assert: structure
    assert isinstance(result, dict)

    # Representative keys that should exist based on specifications file
    expected_keys_present = [
        "adx_15m__window",
        "atr_1d__window",
        "bollinger_bands_1d__window_dev",
        "cmf_4h__window",
        "rsi_1h__window",
        "smc_1d__swing_length",
        "sma_volume_1h__window",
        # EMA and MACD keys
        "ema_15m__fast",
        "ema_15m__mid",
        "ema_15m__slow",
        "macd_15m__window_fast",
        "macd_15m__window_slow",
        "macd_15m__window_sign",
    ]

    for k in expected_keys_present:
        assert k in result, f"Missing expected key: {k}"

    # Check EMA constraints were applied in trial calls for a concrete timeframe (15m)
    # Find calls for EMA mid and slow and ensure their lower bounds use prior returns + 1
    mid_call = next(
        (
            (args, kwargs)
            for (args, kwargs) in call_log_int
            if args[0] == "ema_15m__mid"
        ),
        None,
    )
    slow_call = next(
        (
            (args, kwargs)
            for (args, kwargs) in call_log_int
            if args[0] == "ema_15m__slow"
        ),
        None,
    )
    assert mid_call is not None, "EMA mid suggestion was not called"
    assert slow_call is not None, "EMA slow suggestion was not called"

    # mid lower bound should be at least fast + 1 per implementation
    fast_return = int_returns["ema_15m__fast"]
    assert (
        mid_call[0][1] >= fast_return + 1
    ), f"EMA mid low bound {mid_call[0][1]} should be >= fast+1 ({fast_return+1})"

    # slow lower bound should be at least mid + 1
    mid_return = int_returns["ema_15m__mid"]
    assert (
        slow_call[0][1] >= mid_return + 1
    ), f"EMA slow low bound {slow_call[0][1]} should be >= mid+1 ({mid_return+1})"

    # Check MACD constraint for 15m: slow low bound >= fast_return + 1
    macd_slow_call = next(
        (
            (args, kwargs)
            for (args, kwargs) in call_log_int
            if args[0] == "macd_15m__window_slow"
        ),
        None,
    )
    assert macd_slow_call is not None, "MACD slow suggestion was not called"
    macd_fast_return = int_returns["macd_15m__window_fast"]
    assert macd_slow_call[0][1] >= macd_fast_return + 1, (
        f"MACD slow low bound {macd_slow_call[0][1]} should be >= fast+1 "
        f"({macd_fast_return+1})"
    )


def test_get_model_optimization_space(mocker: pytest.MonkeyPatch) -> None:
    # Arrange: mock trial with typed returns
    trial = mocker.Mock()
    # For floats return a fixed float, for ints a fixed int
    trial.suggest_float.side_effect = lambda name, low, high, **kw: float(low)
    trial.suggest_int.side_effect = lambda name, low, high, **kw: int(low)

    # Act
    params = get_model_optimization_space(trial)

    # Assert: keys and types
    assert set(params.keys()) == {
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "iterations",
    }
    assert isinstance(params["learning_rate"], float)
    assert isinstance(params["l2_leaf_reg"], float)
    assert isinstance(params["depth"], int)
    assert isinstance(params["iterations"], int)
