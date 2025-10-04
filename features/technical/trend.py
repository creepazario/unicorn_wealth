"""Technical trend indicators.

This module implements pure feature functions for trend indicators as
specified in specifications/Unicorn_Wealth_Feature_Set.json.

Functions here are side-effect free: they operate only on their inputs and
return a pandas Series with the computed values.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
from ta.trend import ADXIndicator

__all__ = ["adx_15m", "adx_1h", "adx_4h", "adx_1d"]


def _get_window_from_settings(settings: Dict, path: list[str]) -> int:
    cur: Dict = settings  # type: ignore[assignment]
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(
                f"Missing settings path: {'/'.join(path)}; ensure configuration provides ADX window"
            )
        cur = cur[key]  # type: ignore[assignment]
    try:
        return int(cur)
    except Exception as exc:
        raise ValueError(
            f"ADX window must be an integer at {'/'.join(path)}; got {cur!r}"
        ) from exc


def adx_15m(ohlcv_df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """Compute 15m ADX and directional indicators as a DataFrame.

    Uses ta.trend.ADXIndicator and returns a DataFrame with columns:
    - 'adx'
    - 'adx_pos' (a.k.a. +DI)
    - 'adx_neg' (a.k.a. -DI)
    """
    if not isinstance(ohlcv_df, pd.DataFrame):
        raise TypeError("ohlcv_df must be a pandas DataFrame")
    required = {"high", "low", "close"}
    missing = required - set(ohlcv_df.columns)
    if missing:
        raise KeyError(f"ohlcv_df missing required columns for ADX: {sorted(missing)}")

    window = _get_window_from_settings(settings, ["15m", "adx_15m", "window"])

    indicator = ADXIndicator(
        high=ohlcv_df["high"],
        low=ohlcv_df["low"],
        close=ohlcv_df["close"],
        window=window,
    )
    df = pd.DataFrame(
        {
            "adx": indicator.adx(),
            "adx_pos": indicator.adx_pos(),
            "adx_neg": indicator.adx_neg(),
        }
    )
    return df


def adx_1h(ohlcv_df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """Compute 1h ADX (+/-DI) as a DataFrame."""
    if not isinstance(ohlcv_df, pd.DataFrame):
        raise TypeError("ohlcv_df must be a pandas DataFrame")
    required = {"high", "low", "close"}
    missing = required - set(ohlcv_df.columns)
    if missing:
        raise KeyError(f"ohlcv_df missing required columns for ADX: {sorted(missing)}")

    window = _get_window_from_settings(settings, ["1h", "adx_1h", "window"])

    indicator = ADXIndicator(
        high=ohlcv_df["high"],
        low=ohlcv_df["low"],
        close=ohlcv_df["close"],
        window=window,
    )
    return pd.DataFrame(
        {
            "adx": indicator.adx(),
            "adx_pos": indicator.adx_pos(),
            "adx_neg": indicator.adx_neg(),
        }
    )


def adx_4h(ohlcv_df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """Compute 4h ADX (+/-DI) as a DataFrame."""
    if not isinstance(ohlcv_df, pd.DataFrame):
        raise TypeError("ohlcv_df must be a pandas DataFrame")
    required = {"high", "low", "close"}
    missing = required - set(ohlcv_df.columns)
    if missing:
        raise KeyError(f"ohlcv_df missing required columns for ADX: {sorted(missing)}")

    window = _get_window_from_settings(settings, ["4h", "adx_4h", "window"])

    indicator = ADXIndicator(
        high=ohlcv_df["high"],
        low=ohlcv_df["low"],
        close=ohlcv_df["close"],
        window=window,
    )
    return pd.DataFrame(
        {
            "adx": indicator.adx(),
            "adx_pos": indicator.adx_pos(),
            "adx_neg": indicator.adx_neg(),
        }
    )


def adx_1d(ohlcv_df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """Compute 1d ADX (+/-DI) as a DataFrame."""
    if not isinstance(ohlcv_df, pd.DataFrame):
        raise TypeError("ohlcv_df must be a pandas DataFrame")
    required = {"high", "low", "close"}
    missing = required - set(ohlcv_df.columns)
    if missing:
        raise KeyError(f"ohlcv_df missing required columns for ADX: {sorted(missing)}")

    window = _get_window_from_settings(settings, ["1d", "adx_1d", "window"])

    indicator = ADXIndicator(
        high=ohlcv_df["high"],
        low=ohlcv_df["low"],
        close=ohlcv_df["close"],
        window=window,
    )
    return pd.DataFrame(
        {
            "adx": indicator.adx(),
            "adx_pos": indicator.adx_pos(),
            "adx_neg": indicator.adx_neg(),
        }
    )
