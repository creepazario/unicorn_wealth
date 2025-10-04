from __future__ import annotations

import time
from typing import Any

import ccxt
import pytest

from execution.exchange_clients.base_exchange import BaseExchange
from core.exc import CircuitBreakerOpenError


class DummySettings:
    api_key: str | None = None
    api_secret: str | None = None
    options: dict[str, Any] = {}


class ExchangeUnderTest(BaseExchange):
    """Concrete subclass for testing BaseExchange resilience logic."""

    __test__ = False  # prevent pytest from collecting this helper class

    def __init__(self) -> None:
        super().__init__(DummySettings())
        # Include ccxt.NetworkError as transient so retries occur in tests
        self._transient_exceptions = (
            getattr(ccxt, "NetworkError"),
        ) + self._transient_exceptions
        # Expose a dummy client-like callable we can patch in tests
        self._counter: int = 0

    async def get_balance(self, asset: str) -> Any:  # pragma: no cover - not used
        raise NotImplementedError

    async def place_market_order(
        self, symbol: str, order_side: str, amount: float
    ) -> Any:
        async def _call() -> Any:
            # this function will be monkey-patched in tests
            return None

        return await self._resilient_call(_call)

    async def get_order_book(
        self, symbol: str, limit: int | None = None
    ) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def fetch_ticker(self, symbol: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def cancel_order(
        self, order_id: str, symbol: str | None = None
    ) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover
        return None

    async def set_leverage(
        self, symbol: str, leverage: int
    ) -> None:  # pragma: no cover - test stub
        return None


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold(mocker: Any) -> None:
    ex = ExchangeUnderTest()

    # Make tenacity retry instantly to speed up test and reduce attempts
    mocker.patch("tenacity.nap.sleep", autospec=True, return_value=None)
    ex._retry_max_attempt = 1

    # Patch the internal call to always raise ccxt.NetworkError
    async def failing_call() -> Any:
        raise ccxt.NetworkError("network down")

    # We intercept _resilient_call usage by replacing place_market_order to use our failing_call
    async def place() -> Any:
        return await ex._resilient_call(failing_call)

    # Call repeatedly until breaker opens (threshold from config is 5)
    with pytest.raises(ccxt.NetworkError):
        await place()
    with pytest.raises(ccxt.NetworkError):
        await place()
    with pytest.raises(ccxt.NetworkError):
        await place()
    with pytest.raises(ccxt.NetworkError):
        await place()
    # This call should trigger opening the breaker
    with pytest.raises(ccxt.NetworkError):
        await place()

    # Now subsequent immediate call should raise CircuitBreakerOpenError without attempting call
    with pytest.raises(CircuitBreakerOpenError):
        await place()


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_after_cooldown(mocker: Any) -> None:
    ex = ExchangeUnderTest()

    # Make tenacity retry instantly to speed up test and reduce attempts
    mocker.patch("tenacity.nap.sleep", autospec=True, return_value=None)
    ex._retry_max_attempt = 1

    # Force open the breaker by simulating failures up to threshold
    async def failing_call() -> Any:
        raise ccxt.NetworkError("network down")

    async def place_fail() -> Any:
        return await ex._resilient_call(failing_call)

    for _ in range(5):
        with pytest.raises(ccxt.NetworkError):
            await place_fail()

    # At this moment, CB is open; fast-forward time beyond cooldown
    # Patch time.time to simulate passage of cooldown seconds
    from config import API_CLIENT_SETTINGS

    cooldown = int(API_CLIENT_SETTINGS["CIRCUIT_BREAKER"]["RESET_TIMEOUT"]) + 1
    real_time_time = time.time

    def fake_time() -> float:
        # pretend cooldown seconds have passed since last failure
        return (ex._last_failure_time or real_time_time()) + cooldown

    mocker.patch("time.time", side_effect=fake_time)

    # In half-open state, allow one attempt; make it succeed to reset breaker
    async def success_call() -> str:
        return "ok"

    async def place_success() -> Any:
        return await ex._resilient_call(success_call)

    result = await place_success()
    assert result == "ok"
    # Next call should be allowed again (breaker closed)
    result2 = await place_success()
    assert result2 == "ok"


@pytest.mark.asyncio
async def test_retry_logic_two_failures_then_success(mocker: Any) -> None:
    ex = ExchangeUnderTest()

    # Make tenacity retry instantly
    mocker.patch("tenacity.nap.sleep", autospec=True, return_value=None)

    calls: dict[str, int] = {"n": 0}

    async def flaky_call() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ccxt.NetworkError("temporary issue")
        return "success"

    async def place_flaky() -> Any:
        return await ex._resilient_call(flaky_call)

    result = await place_flaky()
    assert result == "success"
    assert calls["n"] == 3
