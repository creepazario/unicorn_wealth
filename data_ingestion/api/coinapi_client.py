from __future__ import annotations

from typing import Any, Dict, List, Tuple

import aiohttp
import pandas as pd

from data_ingestion.api.base_client import BaseAPIClient


class CoinApiClient(BaseAPIClient):
    """Client for CoinAPI historical endpoints (metrics and OHLCV).

    Usage parameters for fetch_data:
    - endpoint: one of {"metrics", "ohlcv"}. Defaults to "metrics".
    - metric_id: for metrics endpoint, e.g., "DERIVATIVES_FUNDING_RATE_CURRENT".
    - symbol_id: required, e.g., "BINANCEFTS_PERP_ETH_USDT" or "BINANCE_SPOT_BTC_USDT".
    - period_id: interval, e.g., "15MIN".
    - time_start: ISO8601 start timestamp.
    - time_end: optional ISO8601 end timestamp.
    """

    BASE_URL = "https://rest.coinapi.io/v1"

    async def _build_request(
        self, **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:  # noqa: D401
        endpoint = str(kwargs.get("endpoint", "metrics")).lower()
        symbol_id = kwargs["symbol_id"]
        period_id = kwargs["period_id"]
        time_start = kwargs["time_start"]
        time_end = kwargs.get("time_end")

        if endpoint == "ohlcv":
            url = f"{self.BASE_URL}/ohlcv/{symbol_id}/history"
            # Ensure we request the maximum allowed by CoinAPI to avoid pagination truncation
            limit = int(kwargs.get("limit", 100000))
            params: Dict[str, Any] = {
                "period_id": period_id,
                "time_start": time_start,
                "limit": limit,
            }
            if time_end:
                params["time_end"] = time_end
            return url, params

        # metrics
        metric_id = kwargs["metric_id"]
        url = f"{self.BASE_URL}/metrics/symbol/history"
        limit = int(kwargs.get("limit", 100000))
        params = {
            "metric_id": metric_id,
            "symbol_id": symbol_id,
            "period_id": period_id,
            "time_start": time_start,
            "limit": limit,
        }
        if time_end:
            params["time_end"] = time_end
        return url, params

    async def _send_request(
        self, url: str, params: Dict[str, Any]
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        headers = {"X-CoinAPI-Key": self.api_key}
        timeout = aiohttp.ClientTimeout(total=60)
        async with self.session.get(
            url, params=params, headers=headers, timeout=timeout
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _parse_response(self, response: Any) -> pd.DataFrame:  # noqa: D401
        # CoinAPI returns a list for both metrics and OHLCV history endpoints
        if not response:
            return pd.DataFrame(columns=["timestamp"])  # empty
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"CoinAPI error: {response}")

        df = pd.DataFrame(response)
        # Common time field is time_period_start
        if "time_period_start" in df.columns:
            df = df.rename(columns={"time_period_start": "timestamp"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        # For OHLCV, typical fields are price_open/high/low/close and volume_traded
        return df
