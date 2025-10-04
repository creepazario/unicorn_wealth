from __future__ import annotations

from typing import Optional, Sequence, Mapping, Any

import pandas as pd

from utils.time_utils import parse_to_utc

__all__ = ["economic_event_flag", "economic_event_occurs_within_24h"]


def _to_utc_ts(ts: pd.Series | pd.Timestamp | str) -> pd.Series:
    if isinstance(ts, pd.Series):
        return pd.to_datetime(ts, utc=True, errors="coerce")
    return pd.to_datetime(ts, utc=True)


def economic_event_occurs_within_24h(
    events: Sequence[Mapping[str, Any]] | None,
    now: Optional[Any] = None,
) -> bool:
    """Return True if any economic event with impact == 'high' occurs within +/-12 hours of now.

    Note: Function name retained for compatibility, but the logic follows the updated spec.

    Args:
        events: Iterable of event dicts (Finnhub calendar items). Each item should contain
            at least keys 'impact' and 'time' (or 'timestamp').
        now: Optional reference time; if provided can be str/number/datetime and will be
            parsed using utils.time_utils.parse_to_utc. Defaults to current UTC time.

    Returns:
        True if any event has categorical impact 'high' and its time is within
        the inclusive range [now - 12h, now + 12h]; otherwise False.
    """
    if not events:
        return False

    now_utc = (
        parse_to_utc(now) if now is not None else parse_to_utc(pd.Timestamp.utcnow())
    )
    window_start = now_utc - pd.Timedelta(hours=12)
    window_end = now_utc + pd.Timedelta(hours=12)

    for ev in events:
        if not isinstance(ev, Mapping):
            continue
        impact_val = ev.get("impact")
        s = str(impact_val).strip().lower()
        if s != "high":
            continue
        # Support both 'time' and 'timestamp' keys
        ts_value = ev.get("time", ev.get("timestamp"))
        if ts_value is None:
            continue
        try:
            ev_time = parse_to_utc(ts_value)
        except Exception:
            continue
        if window_start <= ev_time <= window_end:
            return True
    return False


def economic_event_flag(
    raw_calendar_df: pd.DataFrame, now: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Compute a boolean flag indicating presence of an economic event with impact 'high' within ±12h of now.

    Rules per spec (operation: economic_event):
    - Use Finnhub economic calendar payload (already parsed to DataFrame with 'timestamp').
    - Exclude the last row of raw incoming data to avoid incomplete data (df.iloc[:-1]).
    - If any row has impact == 'high' (case-insensitive) and the event time is within 12 hours before or after now,
      then set economic_event = True; otherwise False.

    Returns a single-row DataFrame with columns: ["timestamp", "economic_event"], where timestamp is 'now' (UTC).
    """
    # Normalize inputs
    if raw_calendar_df is None or len(raw_calendar_df) == 0:
        base = pd.DataFrame(columns=["timestamp", "economic_event"])  # empty
        return base

    df = raw_calendar_df.copy()

    # Drop potentially incomplete trailing row
    if len(df) > 0:
        df = df.iloc[:-1]

    if len(df) == 0:
        # No rows after dropping -> no event
        ts_now = _to_utc_ts(now or pd.Timestamp.utcnow())
        return pd.DataFrame([[ts_now, False]], columns=["timestamp", "economic_event"])

    # Ensure timestamp is datetime[UTC]
    if "timestamp" not in df.columns:
        # Attempt common fallbacks
        for cand in ("time", "datetime", "date", "time_period_start"):
            if cand in df.columns:
                df = df.rename(columns={cand: "timestamp"})
                break
    df["timestamp"] = _to_utc_ts(df["timestamp"])  # may introduce NaT for bad rows
    df = df[df["timestamp"].notna()].copy()

    # Impact column (categorical strings expected)
    impact = df.get("impact")
    if impact is None:
        impact = pd.Series([None for _ in range(len(df))], index=df.index)

    # Determine now in UTC and ±12h window
    ts_now = _to_utc_ts(now or pd.Timestamp.utcnow())
    window_start = ts_now - pd.Timedelta(hours=12)
    window_end = ts_now + pd.Timedelta(hours=12)
    within_12h = (df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)

    # Strict 'high' impact filter
    impact_str = impact.astype(str).str.lower().str.strip()
    is_high = impact_str.eq("high")

    flag = bool(((within_12h) & (is_high)).any())

    # Return a single-row frame with the decision time as timestamp
    out = pd.DataFrame(
        {
            "timestamp": [ts_now],
            "economic_event": [flag],
        }
    )
    return out
