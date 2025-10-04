from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from features.technical.momentum import rsi_15m
from ta.momentum import RSIIndicator


def _make_synthetic_ohlcv(n: int = 30) -> pd.DataFrame:
    # Create gently trending OHLCV with small oscillation
    base = np.linspace(50, 55, n)
    close = base + np.sin(np.linspace(0, 3 * np.pi, n)) * 0.5
    open_ = close + np.random.default_rng(42).normal(0, 0.1, n)
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    volume = np.full(n, 1000.0)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return df


def test_rsi_15m_calculation():
    # Arrange
    ohlcv = _make_synthetic_ohlcv(30)
    window = 5

    # Act
    result = rsi_15m(ohlcv_df=ohlcv, settings={"15m": {"rsi_15m": {"window": window}}})

    # Assert basic properties
    assert isinstance(result, pd.Series)
    assert len(result) == len(ohlcv)

    # Compute expected using the reference ta implementation
    expected_series = RSIIndicator(close=ohlcv["close"], window=window).rsi()

    # Compare last value
    assert result.iloc[-1] == pytest.approx(
        expected_series.iloc[-1], rel=1e-9, abs=1e-9
    )
