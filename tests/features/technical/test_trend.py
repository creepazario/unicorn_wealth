from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from features.technical.trend import adx_15m
from ta.trend import ADXIndicator


def _make_synthetic_ohlc(n: int = 25) -> pd.DataFrame:
    # Create a simple ascending price series with mild noise for realism
    base = np.linspace(100, 110, n)
    high = pd.Series(base + 1.0)
    low = pd.Series(base - 1.0)
    close = pd.Series(base + np.sin(np.linspace(0, 2 * np.pi, n)) * 0.2)
    return pd.DataFrame({"high": high, "low": low, "close": close})


def test_adx_15m_calculation():
    # Arrange
    df = _make_synthetic_ohlc(30)
    window = 5

    # Act
    result = adx_15m(df["high"], df["low"], df["close"], window=window)

    # Assert basic properties
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)

    # Compute expected using the reference ta implementation
    expected_series = ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=window
    ).adx()

    # Last value should match closely
    assert result.iloc[-1] == pytest.approx(
        expected_series.iloc[-1], rel=1e-9, abs=1e-9
    )
