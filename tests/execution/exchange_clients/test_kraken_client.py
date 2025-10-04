from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from execution.exchange_clients.kraken_client import KrakenClient


class Settings(SimpleNamespace):
    pass


@pytest.mark.asyncio
async def test_kraken_place_market_order_maps_to_ccxt(mocker: Any) -> None:
    mock_client = mocker.AsyncMock()
    mocker.patch(
        "execution.exchange_clients.kraken_client.ccxt.kraken",
        autospec=True,
        return_value=mock_client,
    )

    settings = Settings(api_key="k", api_secret="s", options={})
    client = KrakenClient(settings)

    await client.place_market_order("ETH/USDT", "buy", 0.75)

    mock_client.create_order.assert_awaited_once_with("ETH/USDT", "market", "buy", 0.75)

    await client.close()
