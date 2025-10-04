import pandas as pd
import numpy as np
import pytest

from ta.trend import ADXIndicator

from features.technical.trend import adx_15m, adx_1h, adx_4h, adx_1d


def _make_ohlcv(n_rows: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="h")
    base = np.linspace(100, 200, n_rows)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, size=n_rows)
    close = base + noise
    high = close + np.abs(noise) + 1.0
    low = close - (np.abs(noise) + 1.0)
    volume = rng.integers(100, 1000, size=n_rows)
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
        (adx_15m, ["15m", "adx_15m", "window"], 30),
        (adx_1h, ["1h", "adx_1h", "window"], 30),
        (adx_4h, ["4h", "adx_4h", "window"], 40),
        (adx_1d, ["1d", "adx_1d", "window"], 30),
    ],
)
def test_adx_functions_return_dataframe_with_expected_columns(
    func, settings_path, window
):
    n = window + 100
    df = _make_ohlcv(n)
    settings = {settings_path[0]: {settings_path[1]: {settings_path[2]: window}}}

    result_df = func(df, settings)
    assert isinstance(result_df, pd.DataFrame)
    for col in ["adx", "adx_pos", "adx_neg"]:
        assert col in result_df.columns
    assert len(result_df) == n


@pytest.mark.parametrize(
    "func, settings_path, window",
    [
        (adx_15m, ["15m", "adx_15m", "window"], 30),
        (adx_1h, ["1h", "adx_1h", "window"], 30),
        (adx_4h, ["4h", "adx_4h", "window"], 40),
        (adx_1d, ["1d", "adx_1d", "window"], 30),
    ],
)
def test_adx_matches_ta_library(func, settings_path, window):
    n = window + 120
    df = _make_ohlcv(n)
    settings = {settings_path[0]: {settings_path[1]: {settings_path[2]: window}}}

    result_df = func(df, settings)

    indi = ADXIndicator(df["high"], df["low"], df["close"], window=window)
    expected_adx = indi.adx()
    expected_pos = indi.adx_pos()
    expected_neg = indi.adx_neg()

    pd.testing.assert_series_equal(result_df["adx"], expected_adx, check_names=False)
    pd.testing.assert_series_equal(
        result_df["adx_pos"], expected_pos, check_names=False
    )
    pd.testing.assert_series_equal(
        result_df["adx_neg"], expected_neg, check_names=False
    )
