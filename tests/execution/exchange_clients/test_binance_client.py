from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from execution.exchange_clients.binance_client import BinanceClient


class Settings(SimpleNamespace):
    pass


@pytest.mark.asyncio
async def test_binance_place_market_order_maps_to_ccxt(mocker: Any) -> None:
    # Patch the ccxt async_support alias used in our module: execution.exchange_clients.binance_client.ccxt.binance
    mock_client = mocker.AsyncMock()
    mocker.patch(
        "execution.exchange_clients.binance_client.ccxt.binance",
        autospec=True,
        return_value=mock_client,
    )

    # Instantiate client
    settings = Settings(api_key="k", api_secret="s", options={})
    client = BinanceClient(settings)

    # Call method
    await client.place_market_order("BTC/USDT", "buy", 1.0)

    # Assert mapping to create_order with correct args
    mock_client.create_order.assert_awaited_once_with("BTC/USDT", "market", "buy", 1.0)

    # Close
    await client.close()
