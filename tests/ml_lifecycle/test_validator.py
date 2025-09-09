from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

try:  # Prefer relative import
    from ...ml_lifecycle.validator import BacktestMetrics, ModelValidator
    from ...config import BACKTESTING_SETTINGS
except Exception:  # pragma: no cover - fallback when running tests differently
    from unicorn_wealth.ml_lifecycle.validator import BacktestMetrics, ModelValidator  # type: ignore
    from unicorn_wealth.config import BACKTESTING_SETTINGS  # type: ignore


class _AlwaysLongModel:
    """Simple mock model that always outputs positive predictions."""

    def predict(
        self, X: pd.DataFrame
    ) -> np.ndarray:  # noqa: N802 (external-like signature)
        return np.ones(len(X), dtype=float)


def _make_synthetic_data(n_bars: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create simple 15m OHLCV-like data with steadily increasing prices.

    We design the series multiplicatively so that a 1% TP is reached within the
    same bar after entry, even after accounting for the 0.09% combined costs.

    - open[t] grows by 1% per bar starting at 100.
    - high[t] = open[t] * 1.02; low[t] = open[t] * 0.99; close[t] = open[t] * 1.01
    - features: any numeric columns; content doesn't matter for mock model.

    Using n_bars=5 yields exactly 2 complete trades with the current backtester
    (entries at bars 1 and 3, exits on those same bars).
    """
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="15min")
    opens = 100.0 * (1.01 ** np.arange(n_bars))
    highs = opens * 1.02
    lows = opens * 0.997
    closes = opens * 1.01
    prices = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n_bars),
        },
        index=idx,
    )
    features = pd.DataFrame(
        {
            "feat1": np.arange(n_bars, dtype=float),
            "feat2": np.ones(n_bars, dtype=float) * 2.0,
        },
        index=idx,
    )
    return features, prices


def test_backtest_pnl_calculation():
    # Arrange synthetic data
    features, prices = _make_synthetic_data(n_bars=5)

    # Mock models: only 1h model provided, others missing (NaNs handled)
    models: Dict[str, Any] = {"1h": _AlwaysLongModel()}

    # Instantiate validator (mlflow not used here)
    validator = ModelValidator(
        test_features=features,
        test_prices=prices,
        challenger_run_id="dummy",
    )

    # Act: run backtest
    metrics = validator._run_backtest(models, features, prices)

    # Assert: total PnL positive and equals expected value with costs
    c = (
        BACKTESTING_SETTINGS["SLIPPAGE_PCT"] + BACKTESTING_SETTINGS["TAKER_FEE"]
    ) / 100.0
    tp = 0.01  # from _get_strategic_directive in validator

    # With 5 bars, trades occur on alternating bars: entries at next_open of i=0 and i=2
    next_opens = [prices["open"].iloc[1], prices["open"].iloc[3]]

    # PnL per trade derived from validator pricing logic:
    # entry ep = open_price*(1+c); tp price = ep*(1+tp); exit filled xp = tp*(1-c)
    # pnl = xp - ep = open_price*(1+c)*((1+tp)*(1-c) - 1) = open_price*(1+c)*(tp - c - tp*c)
    per_trade_factor = (1 + c) * (tp - c - tp * c)
    expected_total_pnl = float(
        sum(open_price * per_trade_factor for open_price in next_opens)
    )

    assert metrics.total_pnl > 0.0
    assert pytest.approx(metrics.total_pnl, rel=1e-9, abs=1e-9) == expected_total_pnl
    assert metrics.num_trades == 2


@pytest.mark.parametrize(
    "champion_sharpe, challenger_sharpe, expected",
    [
        # Scenario 1: Challenger promoted (significantly higher Sharpe)
        (1.0, 1.5, True),
        # Scenario 2: Challenger rejected (lower Sharpe)
        (1.0, 0.8, False),
    ],
)
def test_champion_challenger_logic(
    monkeypatch, champion_sharpe, challenger_sharpe, expected
):
    features, prices = _make_synthetic_data(n_bars=5)
    validator = ModelValidator(features, prices, challenger_run_id="dummy")

    # Avoid any real mlflow usage by stubbing loaders and logger
    monkeypatch.setattr(
        ModelValidator, "_load_challenger_models", lambda self, run_id: {"1h": object()}
    )
    monkeypatch.setattr(
        ModelValidator, "_load_champion_models", lambda self: {"1h": object()}
    )
    monkeypatch.setattr(
        ModelValidator, "_log_metrics_to_mlflow", lambda *args, **kwargs: None
    )

    # Mock _run_backtest to return desired Sharpe values
    def _mock_run_backtest(models, *_args, **_kwargs):
        # Decide based on whether models dict is empty or not; we will call twice
        # We can't distinguish champion vs challenger by content here, so we'll patch sequentially
        return BacktestMetrics(
            total_pnl=0.0, sharpe_ratio=0.0, max_drawdown=0.0, num_trades=0
        )

    # We'll patch the method to return different values on subsequent calls
    call_results = [
        BacktestMetrics(0.0, challenger_sharpe, 0.0, 10),
        BacktestMetrics(0.0, champion_sharpe, 0.0, 10),
    ]

    def _sequenced_backtest(self, models, *_args, **_kwargs):  # noqa: ARG001 (unused)
        return call_results.pop(0)

    monkeypatch.setattr(
        ModelValidator, "_run_backtest", _sequenced_backtest, raising=True
    )

    result = validator.validate_models()

    # Compute threshold factor
    threshold_pct = BACKTESTING_SETTINGS["CHAMPION_PROMOTION_THRESHOLD_PCT"] / 100.0
    factor = 1.0 + threshold_pct
    expected_decision = challenger_sharpe >= factor * champion_sharpe

    assert result == expected_decision == expected
