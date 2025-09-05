"""Centralized timestamp parsing and conversion utilities.

All functions in this module are pure and return/operate on timezone-aware
UTC-normalized datetime objects or UNIX timestamps in seconds.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

__all__ = [
    "parse_to_utc",
    "datetime_to_unix",
    "get_current_unix_utc",
]


def parse_to_utc(timestamp: Any) -> datetime:
    """Parse various timestamp representations into a UTC-aware datetime.

    Supported inputs:
    - ISO 8601 strings (e.g., "2025-09-05T04:13:00Z", "2025-09-05T04:13:00+02:00")
    - Numeric seconds since UNIX epoch (int or float)
    - Numeric milliseconds since UNIX epoch (int or float)
    - datetime instances (naive assumed as UTC)

    All outputs are timezone-aware datetime objects normalized to UTC.

    Args:
        timestamp: The input timestamp in one of the supported formats.

    Returns:
        A timezone-aware ``datetime`` object in UTC.

    Raises:
        ValueError: If the input cannot be parsed into a datetime.
    """
    # 1) datetime input
    if isinstance(timestamp, datetime):
        dt = timestamp
        if dt.tzinfo is None:
            # Treat naive datetimes as UTC by convention of this utility
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

    # 2) Numeric input (seconds or milliseconds)
    if isinstance(timestamp, (int, float)):
        value = float(timestamp)
        # Heuristic: values >= 1e12 are considered milliseconds
        # (1e12 ms ~ 2001-09-09 in ms, safely above any current epoch seconds)
        if abs(value) >= 1_000_000_000_000:  # milliseconds
            value /= 1000.0
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError) as exc:  # platform-dependent errors
            raise ValueError(
                f"Numeric timestamp out of valid range: {timestamp}"
            ) from exc

    # 3) String input (ISO 8601 or numeric as string)
    if isinstance(timestamp, str):
        s = timestamp.strip()

        # Numeric-like string? Try to parse as float (sec or ms)
        if s.replace("_", "").replace(".", "", 1).lstrip("-+").isdigit():
            try:
                num = float(s.replace("_", ""))
                return parse_to_utc(num)
            except ValueError:
                pass  # fall through to ISO 8601 parsing

        # Normalize 'Z' suffix to +00:00 for fromisoformat
        if s.endswith("Z") or s.endswith("z"):
            s = s[:-1] + "+00:00"

        # Some ISO strings may have a space separator
        # datetime.fromisoformat handles many variants
        try:
            dt = datetime.fromisoformat(s)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported timestamp string format: {timestamp}"
            ) from exc

        # If no timezone, assume UTC; otherwise convert to UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

    # Unsupported type
    raise ValueError(
        f"Unsupported timestamp type: {type(timestamp).__name__}. "
        "Expected ISO 8601 string, numeric seconds/milliseconds, or datetime."
    )


def datetime_to_unix(dt_object: datetime) -> int:
    """Convert a datetime to an integer UNIX timestamp in seconds (UTC).

    If ``dt_object`` is naive (no tzinfo), it is assumed to be in UTC.

    Args:
        dt_object: A ``datetime`` instance.

    Returns:
        Integer UNIX timestamp in seconds.
    """
    if dt_object.tzinfo is None:
        dt_utc = dt_object.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt_object.astimezone(timezone.utc)
    return int(dt_utc.timestamp())


def get_current_unix_utc() -> int:
    """Return the current UTC time as an integer UNIX timestamp (seconds)."""
    return int(datetime.now(timezone.utc).timestamp())
