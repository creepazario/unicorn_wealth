from __future__ import annotations

from typing import Any, Dict, Tuple

import aiohttp
import pandas as pd

from .base_client import BaseAPIClient


class FinnhubClient(BaseAPIClient):
    """Client for Finnhub economic calendar endpoint.

    fetch_data kwargs:
    - from_date: optional YYYY-MM-DD
    - to_date: optional YYYY-MM-DD
    """

    BASE_URL = "https://finnhub.io/api/v1"

    async def _build_request(
        self, **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:  # noqa: D401
        url = f"{self.BASE_URL}/calendar/economic"
        params: Dict[str, Any] = {}
        if "from" in kwargs:
            params["from"] = kwargs["from"]
        if "to" in kwargs:
            params["to"] = kwargs["to"]
        # Finnhub uses 'token' query param for API key
        if self.api_key:
            params["token"] = self.api_key
        return url, params

    async def _send_request(
        self, url: str, params: Dict[str, Any]
    ) -> Dict[str, Any] | Any:
        timeout = aiohttp.ClientTimeout(total=60)
        async with self.session.get(url, params=params, timeout=timeout) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _parse_response(self, response: Any) -> pd.DataFrame:  # noqa: D401
        if not response:
            return pd.DataFrame(columns=["timestamp"])  # empty
        # Finnhub returns {'economicCalendar': [ ... ]}
        items = response.get("economicCalendar") if isinstance(response, dict) else None
        if not items:
            return pd.DataFrame(columns=["timestamp"])  # safety
        df = pd.DataFrame(items)
        # 'time' -> 'timestamp'
        if "time" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        return df
