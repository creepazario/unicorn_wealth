from __future__ import annotations

from typing import Dict

import pandas as pd
from ta.volatility import AverageTrueRange

__all__ = [
    "atr_15m",
    "atr_1h",
    "atr_4h",
    "atr_1d",
    "atr_normalized_15m",
    "atr_normalized_1h",
    "atr_normalized_4h",
    "atr_normalized_1d",
]


def _get_window_from_settings(settings: Dict, path: list[str]) -> int:
    cur: Dict = settings  # type: ignore[assignment]
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(
                f"Missing settings path: {'/'.join(path)}; ensure configuration provides ATR window"
            )
        cur = cur[key]  # type: ignore[assignment]
    try:
        return int(cur)
    except Exception as exc:
        raise ValueError(
            f"ATR window must be an integer at {'/'.join(path)}; got {cur!r}"
        ) from exc


def atr_15m(ohlcv_df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """Compute 15m ATR DataFrame using ta.volatility.AverageTrueRange.

    Returns the DataFrame produced by AverageTrueRange.average_true_range().
    The output is a DataFrame with a single column 'average_true_range'.
    """
    if not isinstance(ohlcv_df, pd.DataFrame):
        raise TypeError("ohlcv_df must be a pandas DataFrame")
    required = {"high", "low", "close"}
    missing = required - set(ohlcv_df.columns)
    if missing:
        raise KeyError(f"ohlcv_df missing required columns for ATR: {sorted(missing)}")

    window = _get_window_from_settings(settings, ["15m", "atr_15m", "window"])

    atr = AverageTrueRange(
        high=ohlcv_df["high"],
        low=ohlcv_df["low"],
        close=ohlcv_df["close"],
        window=window,
    )
    series = atr.average_true_range()
    df = series.to_frame(name="average_true_range")
    return df


def atr_1h(ohlcv_df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """Compute 1h ATR DataFrame; logic identical to atr_15m."""
    if not isinstance(ohlcv_df, pd.DataFrame):
        raise TypeError("ohlcv_df must be a pandas DataFrame")
    required = {"high", "low", "close"}
    missing = required - set(ohlcv_df.columns)
    if missing:
        raise KeyError(f"ohlcv_df missing required columns for ATR: {sorted(missing)}")

    window = _get_window_from_settings(settings, ["1h", "atr_1h", "window"])

    atr = AverageTrueRange(
        high=ohlcv_df["high"],
        low=ohlcv_df["low"],
        close=ohlcv_df["close"],
        window=window,
    )
    series = atr.average_true_range()
    return series.to_frame(name="average_true_range")


def atr_4h(ohlcv_df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """Compute 4h ATR DataFrame; logic identical to atr_15m."""
    if not isinstance(ohlcv_df, pd.DataFrame):
        raise TypeError("ohlcv_df must be a pandas DataFrame")
    required = {"high", "low", "close"}
    missing = required - set(ohlcv_df.columns)
    if missing:
        raise KeyError(f"ohlcv_df missing required columns for ATR: {sorted(missing)}")

    window = _get_window_from_settings(settings, ["4h", "atr_4h", "window"])

    atr = AverageTrueRange(
        high=ohlcv_df["high"],
        low=ohlcv_df["low"],
        close=ohlcv_df["close"],
        window=window,
    )
    series = atr.average_true_range()
    return series.to_frame(name="average_true_range")


def atr_1d(ohlcv_df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """Compute 1d ATR DataFrame; logic identical to atr_15m."""
    if not isinstance(ohlcv_df, pd.DataFrame):
        raise TypeError("ohlcv_df must be a pandas DataFrame")
    required = {"high", "low", "close"}
    missing = required - set(ohlcv_df.columns)
    if missing:
        raise KeyError(f"ohlcv_df missing required columns for ATR: {sorted(missing)}")

    window = _get_window_from_settings(settings, ["1d", "atr_1d", "window"])

    atr = AverageTrueRange(
        high=ohlcv_df["high"],
        low=ohlcv_df["low"],
        close=ohlcv_df["close"],
        window=window,
    )
    series = atr.average_true_range()
    return series.to_frame(name="average_true_range")


def _atr_normalized_core(atr_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.Series:
    if "average_true_range" not in atr_df.columns:
        raise KeyError("atr_df must contain 'average_true_range' column")
    if "close" not in ohlcv_df.columns:
        raise KeyError("ohlcv_df must contain 'close' column")

    atr_series = atr_df["average_true_range"]
    close_series = ohlcv_df["close"]
    if len(atr_series) != len(close_series):
        try:
            atr_series = atr_series.reindex(ohlcv_df.index)
        except Exception:
            pass

    normalized = (atr_series / (close_series + 1e-9)) * 100.0
    return normalized


def atr_normalized_15m(
    atr_15m_df: pd.DataFrame, ohlcv_15m_df: pd.DataFrame
) -> pd.Series:
    """ATR normalized for 15m timeframe: atr / (close + 1e-9) * 100."""
    s = _atr_normalized_core(atr_15m_df, ohlcv_15m_df)
    s.name = "atr_normalized_15m"
    return s


def atr_normalized_1h(atr_1h_df: pd.DataFrame, ohlcv_1h_df: pd.DataFrame) -> pd.Series:
    """ATR normalized for 1h timeframe: atr / (close + 1e-9) * 100."""
    s = _atr_normalized_core(atr_1h_df, ohlcv_1h_df)
    s.name = "atr_normalized_1h"
    return s


def atr_normalized_4h(atr_4h_df: pd.DataFrame, ohlcv_4h_df: pd.DataFrame) -> pd.Series:
    """ATR normalized for 4h timeframe: atr / (close + 1e-9) * 100."""
    s = _atr_normalized_core(atr_4h_df, ohlcv_4h_df)
    s.name = "atr_normalized_4h"
    return s


def atr_normalized_1d(atr_1d_df: pd.DataFrame, ohlcv_1d_df: pd.DataFrame) -> pd.Series:
    """ATR normalized for 1d timeframe: atr / (close + 1e-9) * 100."""
    s = _atr_normalized_core(atr_1d_df, ohlcv_1d_df)
    s.name = "atr_normalized_1d"
    return s
