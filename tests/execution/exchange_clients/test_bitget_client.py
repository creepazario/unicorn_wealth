from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from execution.exchange_clients.bitget_client import BitgetClient


class Settings(SimpleNamespace):
    pass


@pytest.mark.asyncio
async def test_bitget_place_market_order_maps_to_ccxt(mocker: Any) -> None:
    mock_client = mocker.AsyncMock()
    mocker.patch(
        "execution.exchange_clients.bitget_client.ccxt.bitget",
        autospec=True,
        return_value=mock_client,
    )

    settings = Settings(api_key="k", api_secret="s", options={})
    client = BitgetClient(settings)

    await client.place_market_order("BTC/USDT", "buy", 3.0)

    mock_client.create_order.assert_awaited_once_with("BTC/USDT", "market", "buy", 3.0)

    await client.close()
