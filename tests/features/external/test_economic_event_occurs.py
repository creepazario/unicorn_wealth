from features.external.economic_event import economic_event_occurs_within_24h


def test_positive_high_impact_within_12h():
    now = "2024-01-01T00:00:00Z"
    events = [{"time": "2024-01-01T12:00:00Z", "impact": "high", "event": "CPI"}]
    assert economic_event_occurs_within_24h(events, now=now) is True


def test_negative_high_impact_outside_48h():
    now = "2024-01-01T00:00:00Z"
    events = [{"time": "2024-01-03T00:30:00Z", "impact": "HIGH", "event": "NFP"}]
    # 48h+30m ahead should be False
    assert economic_event_occurs_within_24h(events, now=now) is False


def test_negative_non_high_impact_within_12h():
    now = "2024-01-01T00:00:00Z"
    events = [{"time": "2024-01-01T12:00:00Z", "impact": "medium", "event": "PMI"}]
    assert economic_event_occurs_within_24h(events, now=now) is False
