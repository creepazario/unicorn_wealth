from __future__ import annotations

import abc
import asyncio
import logging
import time
from typing import Any, Dict, Tuple

import aiohttp
import pandas as pd

try:
    import pybreaker  # type: ignore
except Exception:  # pragma: no cover - local shim fallback
    from unicorn_wealth import pybreaker  # type: ignore
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_random_exponential,
    )  # type: ignore
except Exception:  # pragma: no cover - minimal stubs for test env without tenacity

    def retry(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator

    def stop_after_attempt(*args, **kwargs):  # type: ignore
        return None

    def wait_random_exponential(*args, **kwargs):  # type: ignore
        return None


# Load retry/circuit breaker settings from config
try:  # flat import style
    from config import API_CLIENT_SETTINGS  # type: ignore
except Exception:  # pragma: no cover - package import fallback
    try:
        from .. import config as _cfg  # type: ignore

        API_CLIENT_SETTINGS = _cfg.API_CLIENT_SETTINGS  # type: ignore
    except Exception:
        from unicorn_wealth.config import API_CLIENT_SETTINGS  # type: ignore


LOGGER = logging.getLogger(__name__)


class BaseAPIClient(abc.ABC):
    """Abstract base class for historical data API clients.

    Subclasses must implement request construction, sending, and response parsing.
    A single shared aiohttp.ClientSession must be supplied by the caller.
    """

    def __init__(
        self,
        api_key: str | None,
        session: aiohttp.ClientSession,
        *,
        rate_limit_per_sec: float | None = None,
    ) -> None:
        self.api_key = api_key or ""
        self.session = session
        # Simple rate limit state
        self._min_interval = 1.0 / rate_limit_per_sec if rate_limit_per_sec else 0.0
        self._last_request_ts = 0.0

        # Circuit breaker per config
        cb_cfg = API_CLIENT_SETTINGS.get("CIRCUIT_BREAKER", {})
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=int(cb_cfg.get("FAIL_MAX", 5)),
            reset_timeout=int(cb_cfg.get("RESET_TIMEOUT", 60)),
        )

    async def _rate_limit_wait(self) -> None:
        if self._min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        sleep_for = self._min_interval - elapsed
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    # Tenacity retry decorator parameters from config
    _retry_cfg = API_CLIENT_SETTINGS.get("TENACITY_RETRY", {})
    _wait_min = float(_retry_cfg.get("WAIT_MIN", 1))
    _wait_max = float(_retry_cfg.get("WAIT_MAX", 60))
    _stop_attempts = int(_retry_cfg.get("STOP_MAX_ATTEMPT", 5))

    @retry(
        wait=wait_random_exponential(multiplier=1, min=_wait_min, max=_wait_max),
        stop=stop_after_attempt(_stop_attempts),
        reraise=True,
    )
    async def fetch_data(self, **kwargs: Any) -> pd.DataFrame:
        """Orchestrate request/parse with retries and circuit breaker.

        Returns a DataFrame with standardized 'timestamp'.
        """
        url, params = await self._build_request(**kwargs)

        # Rate limit wait before request
        await self._rate_limit_wait()

        # Circuit breaker gating and accounting via try/except
        try:
            # If the breaker is open, raise immediately without recording success
            try:
                is_open_attr = getattr(self._breaker, "is_open", None)
                if callable(is_open_attr):
                    if is_open_attr():
                        raise pybreaker.CircuitBreakerError("Circuit is open")
                else:
                    state = getattr(self._breaker, "state", None)
                    if state is not None:
                        state_open = getattr(pybreaker, "STATE_OPEN", "open")
                        if state == state_open:
                            raise pybreaker.CircuitBreakerError("Circuit is open")
            except pybreaker.CircuitBreakerError as cbe:  # explicitly propagate
                LOGGER.warning(
                    "Circuit breaker OPEN for %s: %s",
                    self.__class__.__name__,
                    cbe,
                )
                raise

            response = await self._send_request(url, params)
            # Mark success with breaker if possible
            try:
                # Mark a successful call by using .call on a trivial function
                self._breaker.call(lambda: True)
            except Exception:  # best-effort; do not mask result
                pass
        except Exception:  # Capture failures to interact with breaker
            # Notify breaker of failure by invoking .call on a function that raises
            try:

                def _raise():
                    raise RuntimeError("downstream failure recorded by breaker")

                self._breaker.call(_raise)
            except pybreaker.CircuitBreakerError:
                # Circuit may transition; ignore here
                pass
            # Re-raise original exception for tenacity to handle
            raise
        finally:
            self._last_request_ts = time.monotonic()

        df = self._parse_response(response)
        if df is None:
            return pd.DataFrame(columns=["timestamp"])  # safety

        # Standardize timestamp column
        if "timestamp" not in df.columns:
            # Try common fields to coerce into 'timestamp'
            for candidate in (
                "time",
                "datetime",
                "date",
                "time_period_start",
                "timestamp",
            ):
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "timestamp"})
                    break
        # Ensure pandas datetime dtype in UTC if possible
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            except Exception:
                pass
        return df

    @abc.abstractmethod
    async def _build_request(self, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        """Return (url, params) for the API call."""

    @abc.abstractmethod
    async def _send_request(
        self, url: str, params: Dict[str, Any]
    ) -> Dict[str, Any] | Any:
        """Perform HTTP request with shared session and return JSON-like response."""

    @abc.abstractmethod
    def _parse_response(self, response: Dict[str, Any] | Any) -> pd.DataFrame:
        """Parse the raw response into a standardized pandas DataFrame."""
