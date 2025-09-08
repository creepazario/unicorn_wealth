from __future__ import annotations

import asyncio
from typing import Any, Dict, Tuple

import pandas as pd
import yfinance as yf

from .base_client import BaseAPIClient


class YFinanceClient(BaseAPIClient):
    """Client for historical market data via yfinance.

    fetch_data kwargs:
    - symbol: Yahoo Finance symbol (e.g., "BTC-USD", "ETH-USD").
    - interval: e.g., "1m", "5m", "15m", "1h", "1d", "1wk", "1mo" (default "1d").
    - start: start date/datetime (str or datetime-like).
    - end: end date/datetime (str or datetime-like).
    """

    async def _build_request(
        self, **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:  # noqa: D401
        symbol = kwargs["symbol"]
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": kwargs.get("interval", "1d"),
            "start": kwargs.get("start"),
            "end": kwargs.get("end"),
        }
        return "yfinance", params

    async def _send_request(
        self, url: str, params: Dict[str, Any]
    ) -> Any:  # noqa: ARG002
        # yfinance is synchronous; run in a worker thread
        def _call() -> pd.DataFrame:
            return yf.download(
                tickers=params["symbol"],
                start=params.get("start"),
                end=params.get("end"),
                interval=params.get("interval"),
                auto_adjust=False,
                prepost=False,
                progress=False,
                threads=False,
            )

        return await asyncio.to_thread(_call)

    def _parse_response(self, response: Any) -> pd.DataFrame:  # noqa: D401
        if response is None:
            return pd.DataFrame(columns=["timestamp"])  # empty
        if isinstance(response, pd.DataFrame):
            df = response.copy()
        else:
            df = pd.DataFrame(response)
        # yfinance returns a Date/Datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            # yfinance uses 'Date' or 'Datetime' depending on interval; if index
            # had no name, pandas uses 'index'
            for cand in ("Date", "Datetime", "index"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "timestamp"})
                    break
        # As a fallback, ensure timestamp column exists
        if "timestamp" not in df.columns:
            for cand in ("Datetime", "Date", "index", "date", "time"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "timestamp"})
                    break
        return df
