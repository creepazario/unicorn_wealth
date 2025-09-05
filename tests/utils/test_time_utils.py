import pytest
from datetime import datetime, timezone

from unicorn_wealth.utils.time_utils import (
    parse_to_utc,
    datetime_to_unix,
)


@pytest.mark.parametrize(
    "input_value",
    [
        "2023-01-01T00:00:00Z",  # ISO 8601 string
        1672531200,  # integer UNIX timestamp in seconds
        1672531200000,  # integer UNIX timestamp in milliseconds
    ],
)
def test_parse_to_utc_various_inputs(input_value):
    expected = datetime(2023, 1, 1, tzinfo=timezone.utc)
    result = parse_to_utc(input_value)
    assert result == expected


def test_datetime_to_unix():
    dt_obj = datetime(2023, 1, 1, tzinfo=timezone.utc)
    result = datetime_to_unix(dt_obj)
    assert result == 1672531200
