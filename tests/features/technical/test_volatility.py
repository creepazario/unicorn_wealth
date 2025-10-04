import pandas as pd
import numpy as np
import pytest

from ta.volatility import AverageTrueRange

from features.technical.volatility import (
    atr_15m,
    atr_1h,
    atr_4h,
    atr_1d,
    atr_normalized_15m,
    atr_normalized_1h,
    atr_normalized_4h,
    atr_normalized_1d,
)


def _make_ohlcv(n_rows: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="h")
    # create a simple synthetic OHLCV where high >= low and close in between
    base = np.linspace(100, 200, n_rows)
    noise = np.random.default_rng(0).normal(0, 1, size=n_rows)
    close = base + noise
    high = close + np.abs(noise) + 1.0
    low = close - (np.abs(noise) + 1.0)
    volume = np.random.default_rng(1).integers(100, 1000, size=n_rows)
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


@pytest.mark.parametrize(
    "func, settings_path, window",
    [
        (atr_15m, ["15m", "atr_15m", "window"], 40),
        (atr_1h, ["1h", "atr_1h", "window"], 40),
        (atr_4h, ["4h", "atr_4h", "window"], 14),
        (atr_1d, ["1d", "atr_1d", "window"], 40),
    ],
)
def test_atr_functions_return_series_with_expected_length(func, settings_path, window):
    n = window + 5
    df = _make_ohlcv(n)
    # Build settings dict with required path
    settings = {settings_path[0]: {settings_path[1]: {settings_path[2]: window}}}

    result_df = func(df, settings)
    assert isinstance(result_df, pd.DataFrame)
    assert "average_true_range" in result_df.columns
    # ta indicators often produce leading NaNs equal to window - 1
    assert len(result_df) == n
    # Ensure it can be converted to Series if needed
    s = result_df["average_true_range"]
    assert isinstance(s, pd.Series)


@pytest.mark.parametrize(
    "func, settings_path, window",
    [
        (atr_15m, ["15m", "atr_15m", "window"], 40),
        (atr_1h, ["1h", "atr_1h", "window"], 40),
        (atr_4h, ["4h", "atr_4h", "window"], 14),
        (atr_1d, ["1d", "atr_1d", "window"], 40),
    ],
)
def test_atr_matches_ta_library(func, settings_path, window):
    n = window + 10
    df = _make_ohlcv(n)
    settings = {settings_path[0]: {settings_path[1]: {settings_path[2]: window}}}

    result_df = func(df, settings)

    ta_atr = AverageTrueRange(df["high"], df["low"], df["close"], window=window)
    expected = ta_atr.average_true_range()

    # Compare allowing for NaNs at the start; use pandas testing
    pd.testing.assert_series_equal(
        result_df["average_true_range"], expected, check_names=False
    )


def _make_simple_atr_and_close():
    atr_df = pd.DataFrame({"average_true_range": [1.0, 2.0, 3.0, 4.0]})
    ohlcv_df = pd.DataFrame({"close": [10.0, 20.0, 40.0, 80.0]})
    # Expected with stabilizer in denominator
    expected = (atr_df["average_true_range"] / (ohlcv_df["close"] + 1e-9)) * 100.0
    return atr_df, ohlcv_df, expected


def test_atr_normalized_15m():
    atr_df, ohlcv_df, expected = _make_simple_atr_and_close()
    result = atr_normalized_15m(atr_df, ohlcv_df)
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected, check_names=False
    )


def test_atr_normalized_1h():
    atr_df, ohlcv_df, expected = _make_simple_atr_and_close()
    result = atr_normalized_1h(atr_df, ohlcv_df)
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected, check_names=False
    )


def test_atr_normalized_4h():
    atr_df, ohlcv_df, expected = _make_simple_atr_and_close()
    result = atr_normalized_4h(atr_df, ohlcv_df)
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected, check_names=False
    )


def test_atr_normalized_1d():
    atr_df, ohlcv_df, expected = _make_simple_atr_and_close()
    result = atr_normalized_1d(atr_df, ohlcv_df)
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected, check_names=False
    )
