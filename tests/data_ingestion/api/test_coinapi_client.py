import asyncio
from typing import Any, Dict

import pandas as pd
import pytest

from unicorn_wealth.data_ingestion.api.coinapi_client import CoinApiClient


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def _no_sleep(duration: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


@pytest.mark.asyncio
async def test_request_formulation_metrics(mocker):
    session = mocker.MagicMock()
    client = CoinApiClient(api_key="KEY123", session=session)

    # Patch _send_request to avoid real HTTP
    mocked = mocker.patch.object(
        client,
        "_send_request",
        return_value=[{"time_period_start": "2024-01-01T00:00:00Z", "v": 1}],
        autospec=True,
    )

    params: Dict[str, Any] = {
        "endpoint": "metrics",
        "metric_id": "DERIVATIVES_FUNDING_RATE_CURRENT",
        "symbol_id": "BINANCEFTS_PERP_ETH_USDT",
        "period_id": "1HRS",
        "time_start": "2024-01-01T00:00:00Z",
        "time_end": "2024-01-02T00:00:00Z",
    }

    df = await client.fetch_data(**params)

    # Assert URL and params passed to _send_request
    assert mocked.call_count == 1
    _, call_args, call_kwargs = mocked.mock_calls[0]
    # call_args: (url, params)
    url, sent_params = call_args
    assert url == "https://rest.coinapi.io/v1/metrics/symbol/history"
    for k in ("metric_id", "symbol_id", "period_id", "time_start", "time_end"):
        assert sent_params[k] == params[k]

    # Response parsing
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]) is True


@pytest.mark.asyncio
async def test_request_formulation_ohlcv_and_parsing(mocker):
    session = mocker.MagicMock()
    client = CoinApiClient(api_key="KEY123", session=session)

    # Mock OHLCV response
    payload = [
        {
            "time_period_start": "2024-01-01T00:00:00Z",
            "price_open": 100.0,
            "price_high": 110.0,
            "price_low": 90.0,
            "price_close": 105.0,
            "volume_traded": 123.45,
        }
    ]
    mocked = mocker.patch.object(
        client, "_send_request", return_value=payload, autospec=True
    )

    df = await client.fetch_data(
        endpoint="ohlcv",
        symbol_id="BINANCE_SPOT_BTC_USDT",
        period_id="1DAY",
        time_start="2024-01-01T00:00:00Z",
    )

    assert mocked.call_count == 1
    _, call_args, _ = mocked.mock_calls[0]
    url, sent_params = call_args
    assert url == "https://rest.coinapi.io/v1/ohlcv/BINANCE_SPOT_BTC_USDT/history"
    assert sent_params == {"period_id": "1DAY", "time_start": "2024-01-01T00:00:00Z"}

    # Parsing
    assert set(
        [
            "timestamp",
            "price_open",
            "price_high",
            "price_low",
            "price_close",
            "volume_traded",
        ]
    ).issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]) is True
