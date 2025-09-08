import asyncio

import pandas as pd
import pytest

from unicorn_wealth.data_ingestion.api.finnhub_client import FinnhubClient


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def _no_sleep(duration: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


@pytest.mark.asyncio
async def test_request_formulation_and_parsing(mocker):
    session = mocker.MagicMock()
    client = FinnhubClient(api_key="FINN_KEY", session=session)

    payload = {
        "economicCalendar": [
            {"time": "2024-01-01T12:30:00Z", "event": "CPI", "actual": 3.2},
            {"time": "2024-01-02T13:00:00Z", "event": "GDP", "actual": 2.1},
        ]
    }
    mocked = mocker.patch.object(
        client, "_send_request", return_value=payload, autospec=True
    )

    df = await client.fetch_data(**{"from": "2024-01-01", "to": "2024-01-02"})

    # Verify request
    assert mocked.call_count == 1
    _, call_args, _ = mocked.mock_calls[0]
    url, params = call_args
    assert url == "https://finnhub.io/api/v1/calendar/economic"
    # token must be included
    assert params["token"] == "FINN_KEY"
    assert params["from"] == "2024-01-01"
    assert params["to"] == "2024-01-02"

    # Parsing
    assert isinstance(df, pd.DataFrame)
    assert set(["timestamp", "event", "actual"]).issubset(df.columns)
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]) is True
