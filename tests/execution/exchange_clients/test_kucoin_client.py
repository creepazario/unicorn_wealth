from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from execution.exchange_clients.kucoin_client import KucoinClient


class Settings(SimpleNamespace):
    pass


@pytest.mark.asyncio
async def test_kucoin_place_market_order_maps_to_ccxt(mocker: Any) -> None:
    mock_client = mocker.AsyncMock()
    mocker.patch(
        "execution.exchange_clients.kucoin_client.ccxt.kucoin",
        autospec=True,
        return_value=mock_client,
    )

    settings = Settings(api_key="k", api_secret="s", options={}, password="p")
    client = KucoinClient(settings)

    await client.place_market_order("BTC/USDT", "buy", 0.33)

    mock_client.create_order.assert_awaited_once_with("BTC/USDT", "market", "buy", 0.33)

    await client.close()
