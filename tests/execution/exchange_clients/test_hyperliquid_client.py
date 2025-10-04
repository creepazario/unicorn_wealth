from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from execution.exchange_clients.hyperliquid_client import HyperLiquidClient


class Settings(SimpleNamespace):
    pass


@pytest.mark.asyncio
async def test_hyperliquid_place_market_order_maps_to_ccxt(mocker: Any) -> None:
    mock_client = mocker.AsyncMock()
    mocker.patch(
        "execution.exchange_clients.hyperliquid_client.ccxt.hyperliquid",
        autospec=True,
        return_value=mock_client,
    )

    settings = Settings(api_key="k", api_secret="s", options={})
    client = HyperLiquidClient(settings)

    await client.place_market_order("BTC/USDT", "sell", 0.1)

    mock_client.create_order.assert_awaited_once_with("BTC/USDT", "market", "sell", 0.1)

    await client.close()
