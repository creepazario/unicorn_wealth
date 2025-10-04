import asyncio
from typing import Any, Dict

import pandas as pd
import pytest

from data_ingestion.api.coinmarketcap_client import CoinMarketCapClient


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def _no_sleep(duration: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


@pytest.mark.asyncio
async def test_request_formulation_quotes(mocker):
    session = mocker.MagicMock()
    client = CoinMarketCapClient(api_key="CMC_KEY", session=session)

    mocked = mocker.patch.object(
        client, "_send_request", return_value={"data": {}}, autospec=True
    )

    params: Dict[str, Any] = {
        "endpoint": "quotes",
        "symbol": "BTC,ETH",
        "time_start": "2024-01-01T00:00:00Z",
        "time_end": "2024-01-02T00:00:00Z",
        "interval": "1h",
        "convert": "USD",
    }

    await client.fetch_data(**params)

    assert mocked.call_count == 1
    _, call_args, _ = mocked.mock_calls[0]
    url, sent_params = call_args
    assert (
        url == "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical"
    )
    for k, v in params.items():
        if k != "endpoint":  # not passed through
            assert sent_params[k] == v


@pytest.mark.asyncio
async def test_parsing_fear_greed(mocker):
    session = mocker.MagicMock()
    client = CoinMarketCapClient(api_key="CMC_KEY", session=session)

    payload = {
        "data": [
            {"timestamp": "2024-01-01T00:00:00Z", "value": 32, "score": "Fear"},
            {"timestamp": "2024-01-02T00:00:00Z", "value": 45, "score": "Neutral"},
        ]
    }
    mocker.patch.object(client, "_send_request", return_value=payload, autospec=True)

    df = await client.fetch_data(endpoint="fear_greed", start=1, limit=2)

    assert isinstance(df, pd.DataFrame)
    assert set(["timestamp", "value", "score"]).issubset(df.columns)
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]) is True
