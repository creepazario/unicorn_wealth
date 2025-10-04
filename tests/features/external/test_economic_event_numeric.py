import pandas as pd

from features.external.economic_event import (
    economic_event_flag,
    economic_event_occurs_within_24h,
)


def test_occurs_numeric_impact_gte7_true():
    now = "2024-01-01T00:00:00Z"
    events = [
        {"time": "2024-01-01T10:00:00Z", "impact": 7},
        {"time": "2024-01-02T00:00:01Z", "impact": 10},
    ]
    assert economic_event_occurs_within_24h(events, now=now) is False


def test_occurs_numeric_impact_lt7_false():
    now = "2024-01-01T00:00:00Z"
    events = [
        {"time": "2024-01-01T10:00:00Z", "impact": 6},
    ]
    assert economic_event_occurs_within_24h(events, now=now) is False


def test_flag_numeric_impact_dataframe_behavior():
    now = pd.Timestamp("2024-01-01T00:00:00Z")
    df = pd.DataFrame(
        [
            {"timestamp": "2023-12-31T00:00:00Z", "impact": 10},  # outside window
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "impact": 7,
            },  # within window, meets threshold
            {"timestamp": "2024-01-05T12:00:00Z", "impact": 9},  # outside window future
            {"timestamp": "2024-01-01T00:00:00Z", "impact": 10},  # trailing row to drop
        ]
    )
    out = economic_event_flag(df, now=now)
    assert bool(out.loc[0, "economic_event"]) is False


def test_flag_numeric_below_threshold_false():
    now = pd.Timestamp("2024-01-01T00:00:00Z")
    df = pd.DataFrame(
        [
            {"timestamp": "2024-01-01T12:00:00Z", "impact": 6},
            {"timestamp": "2024-01-01T00:00:00Z", "impact": 10},  # trailing row to drop
        ]
    )
    out = economic_event_flag(df, now=now)
    assert bool(out.loc[0, "economic_event"]) is False
