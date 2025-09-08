from __future__ import annotations

import asyncio
from typing import Any, Dict, Tuple

import pandas as pd

from .base_client import BaseAPIClient


class SantimentClient(BaseAPIClient):
    """Client for Santiment metrics using the sanpy library.

    fetch_data kwargs:
    - metric: one of metrics listed in the API guide (e.g., "active_addresses_24h").
    - slug: Santiment project slug (e.g., "bitcoin").
    - from_date: ISO date or datetime string.
    - to_date: ISO date or datetime string.
    - interval: e.g., "1d", "1h", "5m".
    """

    async def _build_request(self, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        # sanpy does not use URL; we return a placeholder and pass params
        metric = kwargs["metric"]
        slug = kwargs["slug"]
        from_date = kwargs.get("from_date")
        to_date = kwargs.get("to_date")
        interval = kwargs.get("interval")
        params = {
            "metric": metric,
            "slug": slug,
            "from_date": from_date,
            "to_date": to_date,
            "interval": interval,
        }
        return "sanpy", params

    async def _send_request(
        self, url: str, params: Dict[str, Any]
    ) -> Any:  # noqa: ARG002
        # santiment via sanpy is synchronous; run in a thread
        # sanpy loads API key from env var SANPY_APIKEY / SANTIMENT_API_KEY
        import san as sanpy  # type: ignore

        def _call() -> Any:
            return sanpy.get(
                params["metric"],
                slug=params["slug"],
                from_date=params.get("from_date"),
                to_date=params.get("to_date"),
                interval=params.get("interval"),
            )

        return await asyncio.to_thread(_call)

    def _parse_response(self, response: Any) -> pd.DataFrame:  # noqa: D401
        if response is None:
            return pd.DataFrame(columns=["timestamp"])  # empty
        if isinstance(response, pd.DataFrame):
            df = response.copy()
        else:
            df = pd.DataFrame(response)
        # sanpy commonly returns 'datetime' column
        if "datetime" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        return df
