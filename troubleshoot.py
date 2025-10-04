from __future__ import annotations

import asyncio
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Project imports
from core.dataframe_registry import DataFrameRegistry
from core.config_loader import load_settings
from database.models.raw_data import RawFundingRates


async def main() -> None:
    # Ensure full column display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # Part 1: Check In-Memory DataFrameRegistry
    print("--- In-Memory Check: BTC_funding_rate_30m_df ---")
    registry = DataFrameRegistry()
    try:
        df = await registry.get_df("BTC_funding_rate_30m_df")
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            print(df.tail(5))
        else:
            print("DataFrame not found in registry.")
    except KeyError:
        print("DataFrame not found in registry.")
    except Exception as e:
        print(f"Error accessing registry: {e}")

    # Part 2: Check PostgreSQL Database (raw_funding_rates)
    print("\n--- Database Check: raw_funding_rates table (BTC) ---")
    settings = load_settings()
    database_url = getattr(settings, "database_url", None)
    if not database_url:
        print("DATABASE_URL not configured in settings.")
        return

    # Create async engine and session
    engine = create_async_engine(database_url, future=True, echo=False)
    try:
        async with AsyncSession(engine) as session:
            # Query latest 5 funding rate rows for BTC
            stmt = (
                select(
                    RawFundingRates.timestamp,
                    RawFundingRates.token,
                    RawFundingRates.rate,
                )
                .where(RawFundingRates.token == "BTC")
                .order_by(desc(RawFundingRates.timestamp))
                .limit(5)
            )
            async with session.begin():
                result = await session.execute(stmt)
                rows = result.fetchall()
                cols = result.keys()
                df_db = pd.DataFrame(rows, columns=cols)
                print(df_db)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
