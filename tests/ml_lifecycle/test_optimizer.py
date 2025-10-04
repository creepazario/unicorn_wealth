from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest


def _make_timestamp_series(start: str, end: str, freq: str = "MS") -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=freq, tz="UTC")


def test_wfo_window_generation():
    # Build a dummy dataset spanning 3 years monthly
    ts = _make_timestamp_series("2020-01-01", "2022-12-31", freq="MS")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": range(len(ts)),
            "high": range(len(ts)),
            "low": range(len(ts)),
            "close": range(len(ts)),
            "target_1h": 0,
            "target_4h": 0,
            "target_8h": 0,
            "f1": 1.0,
        }
    )

    # Import here to ensure test isolation
    import config as config
    from ml_lifecycle.optimizer import StrategyOptimizer

    opt = StrategyOptimizer(dataset=df, config_module=config)
    windows = opt._generate_wfo_windows()

    # Using WFO_SETTINGS: train=12m, val=3m, step=3m
    # First window should start at dataset min, end 12m later for train, then 3m val
    assert windows, "Expected at least one WFO window"
    first = windows[0]
    assert first.train_start == pd.Timestamp("2020-01-01T00:00:00Z")
    assert first.train_end == pd.Timestamp("2021-01-01T00:00:00Z")
    assert first.val_start == pd.Timestamp("2021-01-01T00:00:00Z")
    assert first.val_end == pd.Timestamp("2021-04-01T00:00:00Z")

    # Last window's validation end must be <= max timestamp
    last = windows[-1]
    assert last.val_end <= df["timestamp"].max()

    # Compute expected number of windows roughly: start at 2020-01, step by 3 months
    # last window must have val_end <= 2022-12-31
    # We can verify monotonicity and that step difference in train_start is 3 months
    for i in range(1, len(windows)):
        assert windows[i].train_start == windows[i - 1].train_start + pd.DateOffset(
            months=3
        )


def test_objective_function_orchestration(mocker):
    # Build a tiny dataset with OHLCV, features, and targets
    ts = pd.date_range("2021-01-01", periods=200, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "f1": 1.0,
            "target_1h": [1 if i % 2 == 0 else 0 for i in range(len(ts))],
            "target_4h": [0 if i % 2 == 0 else 1 for i in range(len(ts))],
            "target_8h": [1 for _ in range(len(ts))],
        }
    )

    # Import optimizer module and monkeypatch mlflow inside it to avoid side effects
    import config as config
    import ml_lifecycle.optimizer as optimizer_mod

    optimizer_mod.mlflow = None  # disable MLflow in objective

    # Mock DataPreparer in the optimizer module (constructor is called per window)
    DataPreparerMock = mocker.MagicMock()
    optimizer_mod.DataPreparer = DataPreparerMock

    # Mock ModelTrainer: ensure train_models called once per window and returns models dict
    TrainerMockCls = mocker.MagicMock()
    trainer_instance = mocker.MagicMock()
    trainer_instance.train_models.return_value = {
        "1h": object(),
        "4h": object(),
        "8h": object(),
    }
    # Each instantiation returns a fresh instance mock
    TrainerMockCls.side_effect = [trainer_instance, trainer_instance]
    optimizer_mod.ModelTrainer = TrainerMockCls

    # Mock ModelValidator._run_backtest to return controlled sharpe ratios
    # Also mock the constructor to a simple pass-through object with method
    ValidatorMockCls = mocker.MagicMock()
    validator_instance_1 = mocker.MagicMock()
    validator_instance_2 = mocker.MagicMock()
    # Return values for two windows
    validator_instance_1._run_backtest.return_value = SimpleNamespace(sharpe_ratio=1.0)
    validator_instance_2._run_backtest.return_value = SimpleNamespace(sharpe_ratio=3.0)
    ValidatorMockCls.side_effect = [validator_instance_1, validator_instance_2]
    optimizer_mod.ModelValidator = ValidatorMockCls

    # Create optimizer and override its windows to exactly two custom windows
    from ml_lifecycle.optimizer import StrategyOptimizer, WFOWindow

    opt = StrategyOptimizer(
        dataset=df, config_module=config, horizons=("1h", "4h", "8h")
    )

    w1 = WFOWindow(
        train_start=pd.Timestamp("2021-01-01T00:00:00Z"),
        train_end=pd.Timestamp("2021-04-01T00:00:00Z"),
        val_start=pd.Timestamp("2021-04-01T00:00:00Z"),
        val_end=pd.Timestamp("2021-07-01T00:00:00Z"),
    )
    w2 = WFOWindow(
        train_start=pd.Timestamp("2021-04-01T00:00:00Z"),
        train_end=pd.Timestamp("2021-07-01T00:00:00Z"),
        val_start=pd.Timestamp("2021-07-01T00:00:00Z"),
        val_end=pd.Timestamp("2021-10-01T00:00:00Z"),
    )
    opt._wfo_windows = [w1, w2]

    # Set stage to 'models' to use the simpler model optimization space
    opt._stage = "models"

    # Mock optuna.Trial with suggest methods
    trial = SimpleNamespace(
        suggest_float=lambda *args, **kwargs: 0.1,  # noqa: ARG005
        suggest_int=lambda *args, **kwargs: 5,  # noqa: ARG005
        number=0,
    )

    # Act
    avg = opt._objective(trial)  # type: ignore[arg-type]

    # Assert: each component orchestrated per window (2 windows)
    assert DataPreparerMock.call_count == 2
    assert TrainerMockCls.call_count == 2
    assert trainer_instance.train_models.call_count == 2
    assert ValidatorMockCls.call_count == 2
    assert validator_instance_1._run_backtest.call_count == 1
    assert validator_instance_2._run_backtest.call_count == 1

    # Average of 1.0 and 3.0 is 2.0
    assert pytest.approx(avg, rel=1e-9) == 2.0
