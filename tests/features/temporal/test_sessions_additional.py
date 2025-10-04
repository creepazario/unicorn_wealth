import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.temporal.sessions import compute_session_features  # noqa: E402


def _to_ms(ts: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((ts.view("int64") // 1_000_000).astype("int64"), index=ts)


def test_monday_0830_sample_case():
    # Monday at 08:30 UTC -> within London session and London KZ
    ts = pd.DatetimeIndex([pd.Timestamp("2024-01-08 08:30:00+00:00")])
    ts_ms = _to_ms(ts)
    df = compute_session_features(ts_ms, bar_minutes=15)
    row = df.iloc[0]

    assert bool(row["sess_london_flag"]) is True
    assert bool(row["kz_london_flag"]) is True
    assert int(row["bars_since_kz_start"]) == 6
    assert int(row["bars_to_kz_end"]) == 2
    # Additional sanity checks
    assert bool(row["sess_asia_flag"]) is False
    assert bool(row["sess_ny_flag"]) is False
    assert int(row["bars_since_session_open"]) == 6  # 1.5 hours since 07:00 -> 6 bars
