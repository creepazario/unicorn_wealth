from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from execution.exchange_clients.bybit_client import BybitClient


class Settings(SimpleNamespace):
    pass


@pytest.mark.asyncio
async def test_bybit_place_market_order_maps_to_ccxt(mocker: Any) -> None:
    mock_client = mocker.AsyncMock()
    mocker.patch(
        "execution.exchange_clients.bybit_client.ccxt.bybit",
        autospec=True,
        return_value=mock_client,
    )

    settings = Settings(api_key="k", api_secret="s", options={})
    client = BybitClient(settings)

    await client.place_market_order("BTC/USDT", "sell", 2.5)

    mock_client.create_order.assert_awaited_once_with("BTC/USDT", "market", "sell", 2.5)

    await client.close()
