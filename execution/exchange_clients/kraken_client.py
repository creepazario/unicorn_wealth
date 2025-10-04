from __future__ import annotations

from typing import Any

import ccxt.async_support as ccxt

from execution.exchange_clients.base_exchange import BaseExchange


class KrakenClient(BaseExchange):
    """Kraken exchange client using ccxt in async mode (pro)."""

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        transient: tuple[type[BaseException], ...] = (getattr(ccxt, "NetworkError"),)
        self._transient_exceptions = transient + self._transient_exceptions

        api_key = getattr(settings, "api_key", None) or getattr(
            settings, "API_KEY", None
        )
        secret = getattr(settings, "api_secret", None) or getattr(
            settings, "API_SECRET", None
        )
        options = getattr(settings, "options", None) or {}

        self.client = ccxt.kraken(
            {
                "apiKey": api_key,
                "secret": secret,
                "options": options,
            }
        )

    async def get_balance(self, asset: str) -> Any:
        async def _call() -> Any:
            balances = await self.client.fetch_balance()
            total = balances.get("total", {})
            return total.get(asset) or 0

        return await self._resilient_call(_call)

    async def place_market_order(
        self, symbol: str, order_side: str, amount: float
    ) -> Any:
        side = order_side.lower()
        assert side in {"buy", "sell"}

        async def _call() -> Any:
            return await self.client.create_order(symbol, "market", side, amount)

        return await self._resilient_call(_call)

    async def get_order_book(self, symbol: str, limit: int | None = None) -> Any:
        async def _call() -> Any:
            if limit is not None:
                return await self.client.fetch_order_book(symbol, limit=limit)
            return await self.client.fetch_order_book(symbol)

        return await self._resilient_call(_call)

    async def fetch_ticker(self, symbol: str) -> Any:
        async def _call() -> Any:
            return await self.client.fetch_ticker(symbol)

        return await self._resilient_call(_call)

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> Any:
        async def _call() -> Any:
            if symbol is None:
                return await self.client.cancel_order(order_id)
            return await self.client.cancel_order(order_id, symbol)

        return await self._resilient_call(_call)

    async def close(self) -> None:
        if hasattr(self.client, "close"):
            await self.client.close()

    async def set_leverage(self, symbol: str, leverage: int):
        async def _call() -> Any:
            return await self.client.set_leverage(leverage, symbol)

        return await self._resilient_call(_call)
