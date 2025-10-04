from __future__ import annotations

import asyncio

import pandas as pd

from core.config_loader import load_settings
from data_ingestion.api.coinapi_client import CoinApiClient


async def main() -> None:
    # Display settings for pandas output
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("Attempting to connect to CoinAPI...")

    # Load API credentials/settings
    settings = load_settings()
    api_key = getattr(settings, "coinapi_api_key", None)
    if not api_key:
        print("--- FAILURE: Could not fetch data from CoinAPI ---")
        print("Reason: coinapi_api_key is not configured in settings/.env")
        return

    # Prepare HTTP client session via CoinApiClient (expects aiohttp session)
    import aiohttp

    async with aiohttp.ClientSession() as session:
        client = CoinApiClient(api_key=api_key, session=session)

        # Use confirmed static time range (UTC)
        time_start = "2025-09-01T00:00:00"
        time_end = "2025-09-02T00:00:00"

        try:
            df = await client.fetch_data(
                endpoint="metrics",
                metric_id="DERIVATIVES_FUNDING_RATE_CURRENT",
                symbol_id="BINANCEFTS_PERP_BTC_USDT",
                period_id="30MIN",
                time_start=time_start,
                time_end=time_end,
            )
            print("--- SUCCESS: Data fetched from CoinAPI ---")
            try:
                print("Head:\n", df.head())
                print("Tail:\n", df.tail())
            except Exception as e:
                print("Note: Received non-DataFrame response; printing raw type/info.")
                print(type(df), e)
        except Exception as e:
            print("--- FAILURE: Could not fetch data from CoinAPI ---")
            print(f"Exception: {e}")


if __name__ == "__main__":
    asyncio.run(main())
