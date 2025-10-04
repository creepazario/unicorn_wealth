import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
import pytest
import httpx

from data_ingestion.api.santiment_client import SantimentClient, Settings, APIError
import config as app_config


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def _no_sleep(duration: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(SANTIMENT_API_KEY="dummy-key")


@pytest.mark.asyncio
async def test_fetch_data_success(mocker, mock_settings):
    # Prepare mocked httpx AsyncClient
    session = httpx.AsyncClient()

    # Mock response JSON structure as Santiment GraphQL would return
    sample_points = [
        {"datetime": "2024-01-01T00:00:00Z", "value": 1.23},
        {"datetime": "2024-01-01T01:00:00Z", "value": 2.34},
    ]
    graph_response = {
        "data": {
            "mvrv": {
                "timeseriesData": sample_points,
            }
        }
    }

    async def fake_post(
        url: str, json: Dict[str, Any], headers: Dict[str, str], timeout: float
    ):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return graph_response

        return Resp()

    mocker.patch.object(session, "post", side_effect=fake_post)

    client = SantimentClient(session=session, settings=mock_settings)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    df = await client.fetch_data(
        metric_name="mvrv",
        token_slug="ethereum",
        start_date=start,
        end_date=end,
        interval="1h",
    )

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["timestamp", "mvrv"]
    assert len(df) == len(sample_points)
    # Timestamps are datetime and tz-aware UTC
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]) is True
    assert all(ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) == timezone.utc.utcoffset(ts) for ts in df["timestamp"])  # type: ignore[attr-defined]

    await session.aclose()


@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_failures(mocker, mock_settings):
    session = httpx.AsyncClient()

    # Always raise a RequestError on post
    async def raise_request_error(*args, **kwargs):
        raise httpx.RequestError("network issue")

    mocker.patch.object(session, "post", side_effect=raise_request_error)

    client = SantimentClient(session=session, settings=mock_settings)

    # Number of failures to exceed threshold
    max_failures = int(app_config.API_CLIENT_SETTINGS["CIRCUIT_BREAKER_MAX_FAILURES"])

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    # Fail repeatedly; fetch_data is decorated with tenacity and will raise after retries.
    # We count top-level calls; each call triggers a RetryError from tenacity, which our
    # retry_error_callback wraps into APIError. We catch and continue to increment breaker state.
    for _ in range(max_failures):
        with pytest.raises(APIError):
            await client.fetch_data(
                metric_name="mvrv",
                token_slug="ethereum",
                start_date=start,
                end_date=end,
                interval="1h",
            )

    # Now circuit should be open; next call should raise immediately (without attempting post)
    with pytest.raises(APIError):
        await client.fetch_data(
            metric_name="mvrv",
            token_slug="ethereum",
            start_date=start,
            end_date=end,
            interval="1h",
        )

    await session.aclose()
