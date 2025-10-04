from __future__ import annotations

import asyncio
import functools
import time
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.exc import CircuitBreakerOpenError
from config import API_CLIENT_SETTINGS

T = TypeVar("T")


class BaseExchange(ABC):
    """Abstract base class for exchange clients with built-in resilience.

    Implements a simple circuit breaker and retry mechanism. Subclasses should
    implement async methods for actual exchange operations and decorate them
    with `@BaseExchange.resilient()` or use `self._resilient_call` wrapper.
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings

        cb_cfg = API_CLIENT_SETTINGS.get("CIRCUIT_BREAKER", {})
        self._failure_threshold: int = int(cb_cfg.get("FAIL_MAX", 5))
        self._cooldown_period: int = int(cb_cfg.get("RESET_TIMEOUT", 60))

        self._circuit_breaker_open: bool = False
        self._half_open_trial: bool = False
        self._failure_count: int = 0
        self._last_failure_time: Optional[float] = None

        retry_cfg = API_CLIENT_SETTINGS.get("TENACITY_RETRY", {})
        self._retry_wait_min: int = int(retry_cfg.get("WAIT_MIN", 1))
        self._retry_wait_max: int = int(retry_cfg.get("WAIT_MAX", 60))
        self._retry_max_attempt: int = int(retry_cfg.get("STOP_MAX_ATTEMPT", 5))

        # Subclasses may override this to include their transient exception classes
        self._transient_exceptions: tuple[type[BaseException], ...] = (TimeoutError,)

        # Asyncio lock to serialize breaker state updates
        self._cb_lock = asyncio.Lock()

    # -------------------- Circuit Breaker Helpers --------------------
    async def _pre_call_check_or_raise(self) -> None:
        async with self._cb_lock:
            if self._circuit_breaker_open:
                now = time.time()
                assert self._last_failure_time is not None
                elapsed = now - self._last_failure_time
                if elapsed < self._cooldown_period and not self._half_open_trial:
                    raise CircuitBreakerOpenError()
                # allow one half-open trial
                self._half_open_trial = True

    async def _on_call_success(self) -> None:
        async with self._cb_lock:
            self._failure_count = 0
            self._circuit_breaker_open = False
            self._half_open_trial = False
            self._last_failure_time = None

    async def _on_call_failure(self) -> None:
        async with self._cb_lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold:
                self._circuit_breaker_open = True
                # keep half-open flag false until cooldown over
                self._half_open_trial = False

    # -------------------- Resilience Decorator --------------------
    def resilient(
        self,
    ) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
        """Decorator to add circuit breaker and retry around async API calls."""

        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            # Build tenacity retry decorator lazily to capture current settings
            tenacity_retry = retry(
                retry=retry_if_exception_type(self._transient_exceptions),
                wait=wait_exponential(
                    multiplier=self._retry_wait_min, max=self._retry_wait_max
                ),
                stop=stop_after_attempt(self._retry_max_attempt),
                reraise=True,
            )

            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                await self._pre_call_check_or_raise()

                @tenacity_retry
                async def do_call() -> T:
                    return await func(*args, **kwargs)

                try:
                    result = await do_call()
                except Exception:
                    await self._on_call_failure()
                    raise
                else:
                    await self._on_call_success()
                    return result

            return wrapper

        return decorator

    # Optionally provide a programmatic wrapper use
    async def _resilient_call(self, coro_factory: Callable[[], Awaitable[T]]) -> T:
        await self._pre_call_check_or_raise()

        tenacity_retry = retry(
            retry=retry_if_exception_type(self._transient_exceptions),
            wait=wait_exponential(
                multiplier=self._retry_wait_min, max=self._retry_wait_max
            ),
            stop=stop_after_attempt(self._retry_max_attempt),
            reraise=True,
        )

        @tenacity_retry
        async def do_call() -> T:
            return await coro_factory()

        try:
            result = await do_call()
        except Exception:
            await self._on_call_failure()
            raise
        else:
            await self._on_call_success()
            return result

    # -------------------- Abstract API Methods --------------------
    @abstractmethod
    async def get_balance(self, asset: str) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    async def place_market_order(
        self, symbol: str, order_side: str, amount: float
    ) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    async def get_order_book(
        self, symbol: str, limit: int | None = None
    ) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(
        self, order_id: str, symbol: str | None = None
    ) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    async def set_leverage(
        self, symbol: str, leverage: int
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError
