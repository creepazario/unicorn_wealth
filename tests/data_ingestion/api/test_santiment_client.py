import asyncio
from typing import Any, Dict

import pandas as pd
import pytest

from unicorn_wealth.data_ingestion.api.santiment_client import SantimentClient


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def _no_sleep(duration: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


@pytest.mark.asyncio
async def test_request_formulation(mocker):
    session = mocker.MagicMock()
    client = SantimentClient(api_key=None, session=session)

    mocked = mocker.patch.object(
        client,
        "_send_request",
        return_value=pd.DataFrame({"datetime": ["2024-01-01T00:00:00Z"], "v": [1]}),
        autospec=True,
    )

    params: Dict[str, Any] = {
        "metric": "active_addresses_24h",
        "slug": "bitcoin",
        "from_date": "2024-01-01",
        "to_date": "2024-01-10",
        "interval": "1d",
    }

    df = await client.fetch_data(**params)

    assert mocked.call_count == 1
    _, call_args, _ = mocked.mock_calls[0]
    url, sent_params = call_args
    assert url == "sanpy"
    for k, v in params.items():
        assert sent_params[k] == v

    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]) is True


@pytest.mark.asyncio
async def test_response_parsing_from_dict(mocker):
    session = mocker.MagicMock()
    client = SantimentClient(api_key=None, session=session)

    payload = [{"datetime": "2024-01-01T01:00:00Z", "value": 123}]
    mocker.patch.object(client, "_send_request", return_value=payload, autospec=True)

    df = await client.fetch_data(metric="mvrv", slug="ethereum", interval="1h")

    assert "timestamp" in df.columns
    assert df.shape[0] == 1
    assert pd.to_datetime(df.loc[0, "timestamp"]).tzinfo is not None
