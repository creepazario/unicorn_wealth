from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import aiohttp
import pandas as pd

from .base_client import BaseAPIClient


class CoinMarketCapClient(BaseAPIClient):
    """Client for CoinMarketCap historical endpoints.

    fetch_data kwargs:
    - endpoint: one of {"quotes", "global_metrics", "fear_greed"} (default: "quotes").
    - id: comma-separated CMC ids (optional for quotes).
    - symbol: comma-separated symbols (optional for quotes).
    - time_start, time_end: ISO8601 or unix.
    - count: int.
    - interval: str like "5m", "1h", "1d".
    - convert / convert_id / aux / skip_invalid: passthrough.
    """

    BASE_URL = "https://pro-api.coinmarketcap.com"

    async def _build_request(
        self, **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:  # noqa: D401
        endpoint = str(kwargs.get("endpoint", "quotes")).lower()
        params: Dict[str, Any] = {}
        if endpoint == "global_metrics":
            url = f"{self.BASE_URL}/v1/global-metrics/quotes/historical"
            for key in (
                "time_start",
                "time_end",
                "count",
                "interval",
                "convert",
                "convert_id",
                "aux",
            ):
                if key in kwargs and kwargs[key] is not None:
                    params[key] = kwargs[key]
            return url, params
        if endpoint == "fear_greed":
            url = f"{self.BASE_URL}/v3/fear-and-greed/historical"
            for key in ("start", "limit"):
                if key in kwargs and kwargs[key] is not None:
                    params[key] = kwargs[key]
            return url, params

        # default: quotes historical v2
        url = f"{self.BASE_URL}/v2/cryptocurrency/quotes/historical"
        for key in (
            "id",
            "symbol",
            "time_start",
            "time_end",
            "count",
            "interval",
            "convert",
            "convert_id",
            "aux",
            "skip_invalid",
        ):
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        return url, params

    async def _send_request(
        self, url: str, params: Dict[str, Any]
    ) -> Dict[str, Any] | Any:
        headers = {"X-CMC_PRO_API_KEY": self.api_key}
        timeout = aiohttp.ClientTimeout(total=60)
        async with self.session.get(
            url, params=params, headers=headers, timeout=timeout
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _parse_response(self, response: Any) -> pd.DataFrame:  # noqa: D401
        if not response:
            return pd.DataFrame(columns=["timestamp"])  # empty
        data = response.get("data") if isinstance(response, dict) else None
        if data is None:
            return pd.DataFrame(columns=["timestamp"])  # safety

        # Try to handle common shapes for CMC endpoints
        rows: List[Dict[str, Any]] = []

        # Shape A: quotes historical v2 may return dict mapping symbol/id
        # to a list of quotes
        if isinstance(data, dict):
            for key, value in data.items():
                items: Iterable[Any]
                if isinstance(value, dict) and "quotes" in value:
                    items = value.get("quotes", [])
                elif isinstance(value, list):
                    items = value
                else:
                    continue
                for item in items:
                    ts = (
                        item.get("timestamp")
                        or item.get("time_open")
                        or item.get("time")
                    )
                    quote = item.get("quote") or {}
                    # try to pick USD side if present
                    usd = quote.get("USD") if isinstance(quote, dict) else None
                    price = usd.get("price") if isinstance(usd, dict) else None
                    row = {"timestamp": ts, "key": key, "price_usd": price}
                    rows.append(row)

        # Shape B: global metrics/fear & greed often return list under data
        if not rows and isinstance(data, list):
            for item in data:
                ts = item.get("timestamp") or item.get("time") or item.get("date")
                row = {"timestamp": ts}
                # Include primary numeric fields if exist
                for fld in (
                    "value",
                    "score",
                    "market_cap",
                    "total_market_cap",
                    "total_volume_24h",
                ):
                    if fld in item:
                        row[fld] = item[fld]
                # If nested quote
                quote = item.get("quote") if isinstance(item, dict) else None
                if isinstance(quote, dict):
                    usd = quote.get("USD")
                    if isinstance(usd, dict):
                        for k, v in usd.items():
                            row[f"usd_{k}"] = v
                rows.append(row)

        df = pd.DataFrame(rows)
        return df
