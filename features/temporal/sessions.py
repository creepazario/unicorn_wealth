from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "compute_session_features",
]


def _to_utc_datetime(ts_ms: pd.Series) -> pd.DatetimeIndex:
    if isinstance(ts_ms, pd.Series):
        ts = pd.to_datetime(pd.to_numeric(ts_ms, errors="coerce"), unit="ms", utc=True)
        return pd.DatetimeIndex(ts)
    # assume iterable of ints
    ts = pd.to_datetime(
        pd.to_numeric(pd.Series(ts_ms), errors="coerce"), unit="ms", utc=True
    )
    return pd.DatetimeIndex(ts)


def compute_session_features(
    timestamps_ms: pd.Series, bar_minutes: int = 15
) -> pd.DataFrame:
    """Compute sessions/kill-zone flags and bar counters on a 15m base timeline.

    Implements operations:
      - sess_ny_flag, sess_london_flag, sess_asia_flag
      - kz_ny_flag, kz_london_flag
      - friday_flag
      - bars_to_session_close, bars_to_kz_end
      - bars_since_session_open, bars_since_midnight_utc, bars_since_kz_start

    Rules (UTC):
      Sessions:
        - Asia:   23:00–07:00 (wraps midnight)
        - London: 07:00–15:00
        - New York: 15:00–23:00
      Kill-zones:
        - London: 07:00–09:00
        - New York: 15:00–17:00

    Conventions:
      - Flags are booleans (returned as dtype bool then cast by callers as needed).
      - Bar counters are non-negative integers within the window, 0 outside window.
      - bars_since_midnight_utc counts bars since 00:00 UTC of the same day (00:00 => 0).
    """
    if not isinstance(timestamps_ms, pd.Series):
        timestamps_ms = pd.Series(timestamps_ms)

    dt = _to_utc_datetime(timestamps_ms)
    idx = timestamps_ms.index

    hour = dt.hour
    minute = dt.minute
    weekday = dt.weekday  # Monday=0 .. Sunday=6

    # Minutes since midnight
    mins_midnight = hour * 60 + minute

    # Session windows in minutes
    ASIA_START = 23 * 60
    ASIA_END = 7 * 60  # next day
    LON_START = 7 * 60
    LON_END = 15 * 60
    NY_START = 15 * 60
    NY_END = 23 * 60

    # Kill-zone windows in minutes
    LON_KZ_START = 7 * 60
    LON_KZ_END = 9 * 60
    NY_KZ_START = 15 * 60
    NY_KZ_END = 17 * 60

    # Session membership masks with inclusive boundaries (UTC)
    # Use minute-based comparisons to include exact end times
    in_asia = (mins_midnight >= ASIA_START) | (mins_midnight <= ASIA_END)
    in_london = (mins_midnight >= LON_START) & (mins_midnight <= LON_END)
    in_ny = (mins_midnight >= NY_START) & (mins_midnight <= NY_END)

    # Kill-zone membership masks (inclusive of end minute per spec)
    in_kz_london = (mins_midnight >= LON_KZ_START) & (mins_midnight <= LON_KZ_END)
    in_kz_ny = (mins_midnight >= NY_KZ_START) & (mins_midnight <= NY_KZ_END)

    # Flags
    sess_asia_flag = in_asia
    sess_london_flag = in_london
    sess_ny_flag = in_ny
    kz_london_flag = in_kz_london
    kz_ny_flag = in_kz_ny
    friday_flag = weekday == 4

    # Bars since midnight (always non-negative)
    bars_since_midnight = (mins_midnight // bar_minutes).astype("int64")

    # Bars since session open / to close
    # Asia: wraps midnight; two cases for minutes since open and to close
    mins_since_asia_open = np.where(
        hour >= 23, mins_midnight - ASIA_START, mins_midnight + (24 * 60 - ASIA_START)
    )
    mins_to_asia_close = np.where(
        hour >= 23, (24 * 60 - mins_midnight) + ASIA_END, ASIA_END - mins_midnight
    )

    mins_since_london_open = np.clip(mins_midnight - LON_START, 0, None)
    mins_to_london_close = np.clip(LON_END - mins_midnight, 0, None)

    mins_since_ny_open = np.clip(mins_midnight - NY_START, 0, None)
    mins_to_ny_close = np.clip(NY_END - mins_midnight, 0, None)

    # Select session counters by choosing the session with the smallest mins_since_open among active ones
    n = len(dt)
    big = np.full(n, np.inf)

    ms_asia = np.where(in_asia, mins_since_asia_open.astype(float), big)
    ms_lon = np.where(in_london, mins_since_london_open.astype(float), big)
    ms_ny = np.where(in_ny, mins_since_ny_open.astype(float), big)

    mins_since_stack = np.stack([ms_asia, ms_lon, ms_ny], axis=0)
    choice = np.argmin(mins_since_stack, axis=0)
    active_any = in_asia | in_london | in_ny

    # Helper to choose per session
    def choose(arr_asia, arr_lon, arr_ny):
        stack = np.stack([arr_asia, arr_lon, arr_ny], axis=0)
        return stack[choice, np.arange(n)]

    chosen_mins_since = choose(ms_asia, ms_lon, ms_ny)

    mt_asia = np.where(in_asia, mins_to_asia_close.astype(float), big)
    mt_lon = np.where(in_london, mins_to_london_close.astype(float), big)
    mt_ny = np.where(in_ny, mins_to_ny_close.astype(float), big)
    chosen_mins_to_close = choose(mt_asia, mt_lon, mt_ny)

    bars_since_session_open = np.where(
        active_any, np.floor_divide(chosen_mins_since.astype("int64"), bar_minutes), 0
    ).astype("int64")
    bars_to_session_close = np.where(
        active_any, np.ceil(chosen_mins_to_close / bar_minutes).astype("int64"), 0
    ).astype("int64")

    # Kill-zone counters
    mins_since_lon_kz = np.clip(mins_midnight - LON_KZ_START, 0, None)
    mins_to_lon_kz_end = np.clip(LON_KZ_END - mins_midnight, 0, None)

    mins_since_ny_kz = np.clip(mins_midnight - NY_KZ_START, 0, None)
    mins_to_ny_kz_end = np.clip(NY_KZ_END - mins_midnight, 0, None)

    bars_since_kz_start = np.zeros(len(dt), dtype="int64")
    bars_to_kz_end = np.zeros(len(dt), dtype="int64")

    bars_since_kz_start = np.where(
        in_kz_london, (mins_since_lon_kz // bar_minutes), bars_since_kz_start
    )
    bars_since_kz_start = np.where(
        in_kz_ny, (mins_since_ny_kz // bar_minutes), bars_since_kz_start
    )

    bars_to_kz_end = np.where(
        in_kz_london,
        np.ceil(mins_to_lon_kz_end / bar_minutes).astype("int64"),
        bars_to_kz_end,
    )
    bars_to_kz_end = np.where(
        in_kz_ny,
        np.ceil(mins_to_ny_kz_end / bar_minutes).astype("int64"),
        bars_to_kz_end,
    )

    out = pd.DataFrame(
        {
            "sess_asia_flag": pd.Series(sess_asia_flag, index=idx, dtype="boolean"),
            "sess_london_flag": pd.Series(sess_london_flag, index=idx, dtype="boolean"),
            "sess_ny_flag": pd.Series(sess_ny_flag, index=idx, dtype="boolean"),
            "kz_london_flag": pd.Series(kz_london_flag, index=idx, dtype="boolean"),
            "kz_ny_flag": pd.Series(kz_ny_flag, index=idx, dtype="boolean"),
            "friday_flag": pd.Series(friday_flag, index=idx, dtype="boolean"),
            "bars_to_session_close": pd.Series(
                bars_to_session_close, index=idx, dtype="int64"
            ),
            "bars_to_kz_end": pd.Series(bars_to_kz_end, index=idx, dtype="int64"),
            "bars_since_session_open": pd.Series(
                bars_since_session_open, index=idx, dtype="int64"
            ),
            "bars_since_midnight_utc": pd.Series(
                bars_since_midnight, index=idx, dtype="int64"
            ),
            "bars_since_kz_start": pd.Series(
                bars_since_kz_start, index=idx, dtype="int64"
            ),
        }
    )

    # Replace any negatives (shouldn't occur) with 0
    for c in [
        "bars_to_session_close",
        "bars_to_kz_end",
        "bars_since_session_open",
        "bars_since_midnight_utc",
        "bars_since_kz_start",
    ]:
        out[c] = out[c].clip(lower=0)

    return out
