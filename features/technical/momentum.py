"""Technical momentum indicators.

This module implements pure feature functions for momentum indicators as
specified in specifications/Unicorn_Wealth_Feature_Set.json.

Functions here are side-effect free: they operate only on their inputs and
return a pandas Series with the computed values.
"""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator


def rsi_15m(close: pd.Series, window: int) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    Implements the transform described for the `rsi_15m` operation in the
    specifications: ta.momentum.RSIIndicator(close, window=<default from spec>).rsi()

    Parameters
    ----------
    close : pd.Series
        Series of close prices.
    window : int
        Lookback window for the RSI calculation.

    Returns
    -------
    pd.Series
        The RSI values as a pandas Series.
    """
    indicator = RSIIndicator(close=close, window=window)
    return indicator.rsi()
