from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "day_of_week_cos",
    "day_of_week_sin",
    "hour_of_day_cos",
    "hour_of_day_sin",
    "minute_of_hour_cos",
    "minute_of_hour_sin",
]


def _to_utc_datetime(ts_ms: pd.Series) -> pd.Series:
    """Convert an epoch-ms pandas Series to timezone-aware UTC datetime.

    Ensures vectorized conversion and preserves the original index.
    """
    if not isinstance(ts_ms, pd.Series):
        raise TypeError("ts_ms must be a pandas Series of epoch milliseconds")
    # Coerce to numeric, drop invalids as NaT then later numpy trig will yield NaN
    ts_num = pd.to_numeric(ts_ms, errors="coerce")
    return pd.to_datetime(ts_num, unit="ms", utc=True)


def day_of_week_cos(timestamps: pd.Series) -> pd.Series:
    """cos(2*pi*weekday_utc/7) for each UTC timestamp (epoch ms)."""
    dt = _to_utc_datetime(timestamps)
    # Monday=0 .. Sunday=6
    weekday = dt.dt.weekday.astype("float64")
    angle = 2.0 * np.pi * (weekday / 7.0)
    out = np.cos(angle)
    s = pd.Series(out, index=timestamps.index, name="day_of_week_cos", dtype="float64")
    return s


def day_of_week_sin(timestamps: pd.Series) -> pd.Series:
    """sin(2*pi*weekday_utc/7) for each UTC timestamp (epoch ms)."""
    dt = _to_utc_datetime(timestamps)
    weekday = dt.dt.weekday.astype("float64")
    angle = 2.0 * np.pi * (weekday / 7.0)
    out = np.sin(angle)
    s = pd.Series(out, index=timestamps.index, name="day_of_week_sin", dtype="float64")
    return s


def hour_of_day_cos(timestamps: pd.Series) -> pd.Series:
    """cos(2*pi*hour_utc/24) for each UTC timestamp (epoch ms)."""
    dt = _to_utc_datetime(timestamps)
    hour = dt.dt.hour.astype("float64")
    angle = 2.0 * np.pi * (hour / 24.0)
    out = np.cos(angle)
    s = pd.Series(out, index=timestamps.index, name="hour_of_day_cos", dtype="float64")
    return s


def hour_of_day_sin(timestamps: pd.Series) -> pd.Series:
    """sin(2*pi*hour_utc/24) for each UTC timestamp (epoch ms)."""
    dt = _to_utc_datetime(timestamps)
    hour = dt.dt.hour.astype("float64")
    angle = 2.0 * np.pi * (hour / 24.0)
    out = np.sin(angle)
    s = pd.Series(out, index=timestamps.index, name="hour_of_day_sin", dtype="float64")
    return s


def minute_of_hour_cos(timestamps: pd.Series) -> pd.Series:
    """cos(2*pi*minute_utc/60) for each UTC timestamp (epoch ms)."""
    dt = _to_utc_datetime(timestamps)
    minute = dt.dt.minute.astype("float64")
    angle = 2.0 * np.pi * (minute / 60.0)
    out = np.cos(angle)
    s = pd.Series(
        out, index=timestamps.index, name="minute_of_hour_cos", dtype="float64"
    )
    return s


def minute_of_hour_sin(timestamps: pd.Series) -> pd.Series:
    """sin(2*pi*minute_utc/60) for each UTC timestamp (epoch ms)."""
    dt = _to_utc_datetime(timestamps)
    minute = dt.dt.minute.astype("float64")
    angle = 2.0 * np.pi * (minute / 60.0)
    out = np.sin(angle)
    s = pd.Series(
        out, index=timestamps.index, name="minute_of_hour_sin", dtype="float64"
    )
    return s
