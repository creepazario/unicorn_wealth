from __future__ import annotations

from typing import Dict

import pandas as pd
from ta.momentum import RSIIndicator

__all__ = ["rsi_15m"]


def _get_window_from_settings(settings: Dict, path: list[str]) -> int:
    cur: Dict = settings  # type: ignore[assignment]
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(
                f"Missing settings path: {'/'.join(path)}; ensure configuration provides RSI window"
            )
        cur = cur[key]  # type: ignore[assignment]
    try:
        return int(cur)
    except Exception as exc:
        raise ValueError(
            f"RSI window must be an integer at {'/'.join(path)}; got {cur!r}"
        ) from exc


def rsi_15m(ohlcv_df: pd.DataFrame, settings: Dict) -> pd.Series:
    """Compute 15m RSI as a pandas Series using ta library.

    Args:
        ohlcv_df: DataFrame with columns including 'close'. Index alignment is preserved.
        settings: Dict-like settings containing settings['15m']['rsi_15m']['window'].

    Returns:
        pd.Series of RSI values aligned to ohlcv_df index.
    """
    if not isinstance(ohlcv_df, pd.DataFrame):
        raise TypeError("ohlcv_df must be a pandas DataFrame")
    if "close" not in ohlcv_df.columns:
        raise KeyError("ohlcv_df must contain a 'close' column for RSI calculation")

    window = _get_window_from_settings(settings, ["15m", "rsi_15m", "window"])

    indicator = RSIIndicator(close=ohlcv_df["close"], window=window)
    rsi = indicator.rsi()
    rsi.name = "rsi_15m"
    return rsi
