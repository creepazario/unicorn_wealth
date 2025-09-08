from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from features.technical.momentum import rsi_15m
from ta.momentum import RSIIndicator


def _make_synthetic_close(n: int = 30) -> pd.Series:
    # Create a gently trending series with small oscillation
    base = np.linspace(50, 55, n)
    close = pd.Series(base + np.sin(np.linspace(0, 3 * np.pi, n)) * 0.5)
    return close


def test_rsi_15m_calculation():
    # Arrange
    close = _make_synthetic_close(30)
    window = 5

    # Act
    result = rsi_15m(close=close, window=window)

    # Assert basic properties
    assert isinstance(result, pd.Series)
    assert len(result) == len(close)

    # Compute expected using the reference ta implementation
    expected_series = RSIIndicator(close=close, window=window).rsi()

    # Compare last value
    assert result.iloc[-1] == pytest.approx(
        expected_series.iloc[-1], rel=1e-9, abs=1e-9
    )
