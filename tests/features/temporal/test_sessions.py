import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.temporal.sessions import compute_session_features  # noqa: E402


def _to_ms(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((idx.view("int64") // 1_000_000).astype("int64"), index=idx)


def test_london_killzone_flags_and_counters_inclusive_end():
    # Friday 2024-01-05, 06:45 to 09:15 UTC
    idx = pd.date_range(
        "2024-01-05 06:45:00+00:00",
        "2024-01-05 15:00:00+00:00",
        freq="15min",
        inclusive="both",
        tz="UTC",
    )
    ts_ms = _to_ms(idx)
    df = compute_session_features(ts_ms, bar_minutes=15)

    # London session 07:00–15:00 inclusive (15:00 included)
    # London KZ 07:00–09:00 inclusive of 09:00
    # Check some key timestamps
    m = {str(ts): i for i, ts in enumerate(idx)}

    # 06:45 - before London session/KZ, but still in Asia session
    i = m["2024-01-05 06:45:00+00:00"]
    assert not df.iloc[i]["sess_london_flag"]
    assert not df.iloc[i]["kz_london_flag"]
    assert df.iloc[i]["sess_asia_flag"]
    assert df.iloc[i]["bars_since_session_open"] == 31
    assert df.iloc[i]["bars_to_session_close"] == 1
    assert df.iloc[i]["bars_since_kz_start"] == 0
    assert df.iloc[i]["bars_to_kz_end"] == 0

    # 07:00 - start of session and KZ
    i = m["2024-01-05 07:00:00+00:00"]
    assert df.iloc[i]["sess_london_flag"]
    assert df.iloc[i]["kz_london_flag"]
    assert df.iloc[i]["bars_since_session_open"] == 0
    assert df.iloc[i]["bars_to_session_close"] == 32  # 8h left
    assert df.iloc[i]["bars_since_kz_start"] == 0
    assert df.iloc[i]["bars_to_kz_end"] == 8  # 2h KZ

    # 08:45 - within KZ
    i = m["2024-01-05 08:45:00+00:00"]
    assert df.iloc[i]["kz_london_flag"]
    assert df.iloc[i]["bars_since_kz_start"] == 7
    assert df.iloc[i]["bars_to_kz_end"] == 1

    # 09:00 - inclusive end of KZ
    i = m["2024-01-05 09:00:00+00:00"]
    assert df.iloc[i]["kz_london_flag"]
    assert df.iloc[i]["bars_since_kz_start"] == 8
    assert df.iloc[i]["bars_to_kz_end"] == 0

    # 09:15 - after KZ
    i = m["2024-01-05 09:15:00+00:00"]
    assert not df.iloc[i]["kz_london_flag"]
    assert df.iloc[i]["bars_since_kz_start"] == 0
    assert df.iloc[i]["bars_to_kz_end"] == 0

    # 14:45 - last bar in session
    i = m["2024-01-05 14:45:00+00:00"]
    assert df.iloc[i]["sess_london_flag"]
    assert df.iloc[i]["bars_since_session_open"] == 31
    assert df.iloc[i]["bars_to_session_close"] == 1

    # 15:00 - boundary: London inclusive end, NY session start; choose NY for counters
    i = m["2024-01-05 15:00:00+00:00"]
    assert df.iloc[i]["sess_london_flag"]
    assert df.iloc[i]["sess_ny_flag"]
    assert df.iloc[i]["bars_since_session_open"] == 0
    assert df.iloc[i]["bars_to_session_close"] == 32


def test_newyork_killzone_and_friday_flag():
    # Friday 2024-01-12, 14:45 to 17:15 UTC
    idx = pd.date_range(
        "2024-01-12 14:45:00+00:00",
        "2024-01-12 17:15:00+00:00",
        freq="15min",
        inclusive="both",
        tz="UTC",
    )
    ts_ms = _to_ms(idx)
    df = compute_session_features(ts_ms, bar_minutes=15)

    # 15:00 - start of NY session and KZ; also Friday flag should be true
    i = list(idx).index(pd.Timestamp("2024-01-12 15:00:00+00:00"))
    assert df.iloc[i]["sess_ny_flag"]
    assert df.iloc[i]["kz_ny_flag"]
    assert df.iloc[i]["friday_flag"]
    assert df.iloc[i]["bars_since_session_open"] == 0
    assert df.iloc[i]["bars_to_session_close"] == 32
    assert df.iloc[i]["bars_since_kz_start"] == 0
    assert df.iloc[i]["bars_to_kz_end"] == 8

    # 17:00 - inclusive end of KZ
    i = list(idx).index(pd.Timestamp("2024-01-12 17:00:00+00:00"))
    assert df.iloc[i]["kz_ny_flag"]
    assert df.iloc[i]["bars_since_kz_start"] == 8
    assert df.iloc[i]["bars_to_kz_end"] == 0


def test_asia_session_wrap_and_bars_since_midnight():
    # Monday 2024-01-08 (weekday), from 22:45 to 00:30 next day
    idx = pd.date_range(
        "2024-01-08 22:45:00+00:00",
        "2024-01-09 00:30:00+00:00",
        freq="15min",
        inclusive="both",
        tz="UTC",
    )
    ts_ms = _to_ms(idx)
    df = compute_session_features(ts_ms, bar_minutes=15)

    # 23:00 - start of Asia session
    i = list(idx).index(pd.Timestamp("2024-01-08 23:00:00+00:00"))
    assert df.iloc[i]["sess_asia_flag"]
    assert df.iloc[i]["bars_since_session_open"] == 0

    # 00:00 next day - still Asia session; bars_since_midnight_utc should be 0
    i = list(idx).index(pd.Timestamp("2024-01-09 00:00:00+00:00"))
    assert df.iloc[i]["sess_asia_flag"]
    assert df.iloc[i]["bars_since_session_open"] == 4
    assert df.iloc[i]["bars_since_midnight_utc"] == 0

    # 00:30 - bars_since_midnight_utc should be 2
    i = list(idx).index(pd.Timestamp("2024-01-09 00:30:00+00:00"))
    assert df.iloc[i]["bars_since_midnight_utc"] == 2
