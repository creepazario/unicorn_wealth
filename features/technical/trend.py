"""Technical trend indicators.

This module implements pure feature functions for trend indicators as
specified in specifications/Unicorn_Wealth_Feature_Set.json.

Functions here are side-effect free: they operate only on their inputs and
return a pandas Series with the computed values.
"""

from __future__ import annotations

import pandas as pd
from ta.trend import ADXIndicator


def adx_15m(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Compute the Average Directional Movement Index (ADX).

    Implements the transform described for the `adx_15m` operation in the
    specifications: ta.trend.ADXIndicator(high, low, close, window=<default from spec>)

    Parameters
    ----------
    high : pd.Series
        Series of high prices.
    low : pd.Series
        Series of low prices.
    close : pd.Series
        Series of close prices.
    window : int
        Lookback window for the ADX calculation.

    Returns
    -------
    pd.Series
        The ADX values as a pandas Series.
    """
    indicator = ADXIndicator(high=high, low=low, close=close, window=window)
    return indicator.adx()
