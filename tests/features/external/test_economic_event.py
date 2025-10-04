import pandas as pd

from features.external.economic_event import economic_event_flag


def test_economic_event_flag_true_within_24h_and_high_impact():
    now = pd.Timestamp("2024-01-02T12:00:00Z")
    df = pd.DataFrame(
        [
            {"timestamp": "2024-01-01T00:00:00Z", "impact": "medium", "event": "Old"},
            {"timestamp": "2024-01-03T00:00:00Z", "impact": "HIGH", "event": "CPI"},
            # trailing row (should be dropped even if high)
            {"timestamp": "2024-01-02T12:00:00Z", "impact": "high", "event": "DROP_ME"},
        ]
    )

    out = economic_event_flag(df, now=now)
    assert list(out.columns) == ["timestamp", "economic_event"]
    assert len(out) == 1
    assert bool(out.loc[0, "economic_event"]) is True


def test_economic_event_flag_false_no_high_or_outside_window():
    now = pd.Timestamp("2024-01-02T12:00:00Z")
    df = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-10T00:00:00Z",
                "impact": "low",
                "event": "Future far",
            },
            {
                "timestamp": "2023-12-20T00:00:00Z",
                "impact": "medium",
                "event": "Past far",
            },
            {"timestamp": "2024-01-02T12:00:00Z", "impact": "high", "event": "DROP_ME"},
        ]
    )

    out = economic_event_flag(df, now=now)
    assert list(out.columns) == ["timestamp", "economic_event"]
    assert len(out) == 1
    assert bool(out.loc[0, "economic_event"]) is False


def test_handles_missing_timestamp_column_via_time_alias():
    now = pd.Timestamp("2024-01-02T12:00:00Z")
    df = pd.DataFrame(
        [
            {"time": "2024-01-03T00:00:00Z", "impact": "high"},
            {"time": "2024-01-04T00:00:00Z", "impact": "low"},
        ]
    )

    out = economic_event_flag(df, now=now)
    assert bool(out.loc[0, "economic_event"]) is True
