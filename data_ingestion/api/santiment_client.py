from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.time_utils import parse_to_utc
import config as app_config


class APIError(Exception):
    """Raised for API related errors (circuit open or HTTP/GraphQL failures)."""


@dataclass
class Settings:
    """Lightweight settings shim.

    This project does not currently define a Pydantic Settings model. The
    specification requests passing a Pydantic `Settings` object; to keep
    coupling minimal and avoid introducing a new dependency surface here,
    this dataclass acts as a minimal interface adapter. If a Pydantic
    Settings model exists in your application, you can pass it as long as it
    exposes `SANTIMENT_API_KEY` and optionally overrides values in
    `config.API_CLIENT_SETTINGS` via attributes of the same names.
    """

    SANTIMENT_API_KEY: Optional[str] = None


def _retry_error_callback(retry_state):
    # retry_state.args: (self, metric_name, token_slug, start_date, end_date, interval)
    if retry_state.args:
        self_obj = retry_state.args[0]
        try:
            self_obj._record_failure()  # type: ignore[attr-defined]
        except Exception:
            pass
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        raise APIError("Santiment fetch failed after retries with unknown error")
    raise APIError(f"Santiment fetch failed after retries: {exc}") from exc


class SantimentClient:
    """Asynchronous client to query Santiment GraphQL timeseries metrics.

    Implements simple state-based circuit breaker and retry with tenacity.
    """

    _GRAPHQL_ENDPOINT = "https://api.santiment.net/graphql"

    def __init__(self, session: httpx.AsyncClient, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._api_key = getattr(settings, "SANTIMENT_API_KEY", None)

        # Circuit breaker state
        self._circuit_open: bool = False
        self._failure_count: int = 0
        self._last_failure_time: Optional[datetime] = None

        # Load config values (use config.py with sane defaults)
        tenacity_attempts = (
            getattr(settings, "TENACITY_MAX_ATTEMPTS", None)
            or app_config.API_CLIENT_SETTINGS.get("TENACITY_MAX_ATTEMPTS")
            or 3
        )
        wait_multiplier = (
            getattr(settings, "TENACITY_WAIT_MULTIPLIER", None)
            or app_config.API_CLIENT_SETTINGS.get("TENACITY_WAIT_MULTIPLIER")
            or 1
        )
        self._retry_stop = stop_after_attempt(int(tenacity_attempts))
        self._retry_wait = wait_exponential(multiplier=float(wait_multiplier))

        self._cb_max_failures: int = (
            getattr(settings, "CIRCUIT_BREAKER_MAX_FAILURES", None)
            or app_config.API_CLIENT_SETTINGS.get("CIRCUIT_BREAKER_MAX_FAILURES")
            or 5
        )
        self._cb_reset_seconds: int = (
            getattr(settings, "CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS", None)
            or app_config.API_CLIENT_SETTINGS.get(
                "CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS"
            )
            or 60
        )

    async def _post_graphql(
        self, query: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Apikey {self._api_key}"

        resp = await self._session.post(
            self._GRAPHQL_ENDPOINT,
            json={"query": query, "variables": variables},
            headers=headers,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data and data["errors"]:
            raise APIError(f"GraphQL errors: {data['errors']}")
        return data

    def _check_circuit(self) -> None:
        if not self._circuit_open:
            return
        assert self._last_failure_time is not None
        elapsed = datetime.now(timezone.utc) - self._last_failure_time
        if elapsed.total_seconds() < self._cb_reset_seconds:
            raise APIError("Circuit breaker open. Retry later.")
        # Half-open: allow a single trial request
        self._circuit_open = False

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)
        if self._failure_count >= self._cb_max_failures:
            self._circuit_open = True

    def _record_success(self) -> None:
        self._failure_count = 0
        self._circuit_open = False
        self._last_failure_time = None

    @staticmethod
    def _build_query(metric: str) -> str:
        # Santiment GraphQL timeseries query via getMetric(metric: String!)
        # Enforce includeIncompleteData: false to avoid partial rows
        return (
            "query ($metric: String!, $slug: String!, $from: DateTime!, $to: DateTime!, $interval: interval!) {\n"
            "  getMetric(metric: $metric) {\n"
            "    timeseriesData(slug: $slug, from: $from, to: $to, interval: $interval, includeIncompleteData: false) {\n"
            "      datetime\n"
            "      value\n"
            "    }\n"
            "  }\n"
            "}\n"
        )

    @staticmethod
    def _to_dataframe(points: List[Dict[str, Any]], metric_name: str) -> pd.DataFrame:
        # Standardize output to [timestamp, <metric_name>] per tests/specs.
        col = metric_name or "value"
        if not points:
            # Create empty with desired columns
            return pd.DataFrame(columns=["timestamp", col]).astype(
                {"timestamp": "datetime64[ns, UTC]"}
            )

        ts: List[datetime] = [parse_to_utc(p.get("datetime")) for p in points]
        vals: List[Any] = [p.get("value") for p in points]
        df = pd.DataFrame({"timestamp": ts, col: vals})
        # Ensure tz-aware UTC (parse_to_utc already returns UTC-aware)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    @retry(
        wait=wait_exponential(
            multiplier=float(
                app_config.API_CLIENT_SETTINGS.get("TENACITY_WAIT_MULTIPLIER", 1)
            )
        ),
        stop=stop_after_attempt(
            int(app_config.API_CLIENT_SETTINGS.get("TENACITY_MAX_ATTEMPTS", 3))
        ),
        retry=retry_if_exception_type((httpx.HTTPError, APIError)),
        reraise=True,
        retry_error_callback=_retry_error_callback,
    )
    async def fetch_data(
        self,
        metric: Optional[str] = None,
        slug: Optional[str] = None,
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        # Backward-compat aliases
        metric_name: Optional[str] = None,
        token_slug: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch a Santiment metric time-series for a token.

        Args accept both new names (metric, slug) and legacy names (metric_name, token_slug).
        If start_date/end_date are not provided, they are inferred from interval:
          - 15m: last 10 days; 1d: last 120 days; else: last 90 days.

        Returns a DataFrame with columns: [timestamp, value].
        """
        # Circuit breaker pre-check
        self._check_circuit()

        # Resolve parameter aliases
        metric_resolved = metric or metric_name
        slug_resolved = slug or token_slug
        if not metric_resolved or not slug_resolved:
            raise APIError("SantimentClient.fetch_data requires metric and slug")

        # Default time window if not provided
        from datetime import timedelta

        now_utc = datetime.now(timezone.utc)
        if end_date is None:
            end_date = now_utc
        if start_date is None:
            if interval.lower() in ("15m", "15min", "15MIN", "15Min"):
                start_date = end_date - timedelta(days=10)
            elif interval.lower() in ("1d", "1day", "D"):
                start_date = end_date - timedelta(days=120)
            else:
                start_date = end_date - timedelta(days=90)

        # Normalize input datetimes to UTC ISO strings
        start_iso = parse_to_utc(start_date).isoformat().replace("+00:00", "Z")
        end_iso = parse_to_utc(end_date).isoformat().replace("+00:00", "Z")

        query = self._build_query(metric_resolved)
        variables = {
            "metric": metric_resolved,
            "slug": slug_resolved,
            "from": start_iso,
            "to": end_iso,
            "interval": interval,
        }

        data = await self._post_graphql(query, variables)

        # Success path
        self._record_success()

        # Extract points. Prefer GraphQL shape data.getMetric.timeseriesData,
        # but accept tests' alternate shape data.<metric>.timeseriesData as well.
        timeseries: List[Dict[str, Any]] = []
        data_block = data.get("data", {}) or {}
        metric_block = data_block.get("getMetric")
        if isinstance(metric_block, dict) and "timeseriesData" in metric_block:
            timeseries = metric_block.get("timeseriesData") or []
        else:
            # Fallback: find first dict under data that contains timeseriesData
            for v in data_block.values():
                if isinstance(v, dict) and "timeseriesData" in v:
                    timeseries = v.get("timeseriesData") or []
                    break

        return self._to_dataframe(timeseries, metric_resolved)
