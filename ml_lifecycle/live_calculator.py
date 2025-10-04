from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Dict

import pandas as pd

from core.dataframe_registry import DataFrameRegistry
from features.feature_engine import UnifiedFeatureEngine


class LiveFeatureCalculator:
    """Runs live feature calculations in background processes.

    Offloads CPU-bound feature engineering to a ProcessPoolExecutor to avoid
    blocking the main asyncio event loop that manages I/O (e.g., live data streams).
    """

    def __init__(
        self,
        engine: UnifiedFeatureEngine,
        registry: DataFrameRegistry,
        process_pool: ProcessPoolExecutor,
    ) -> None:
        self._engine = engine
        self._registry = registry
        self._pool = process_pool

    async def calculate_all_horizons(
        self, tokens: list[str]
    ) -> Dict[str, pd.DataFrame]:
        """Compute the feature pipeline once, then assemble subsets per horizon.

        - Offloads a single run_15m_pipeline call to the process pool.
        - Splits columns according to config.FEATURE_PARAMS for each horizon.
        - Returns a dict mapping horizon to a single-row DataFrame.
        """
        from config import FEATURE_PARAMS as FP

        loop = asyncio.get_running_loop()
        # Choose representative token (first)
        token = tokens[0] if tokens else None
        if not token:
            return {h: pd.DataFrame() for h in ("1h", "4h", "8h")}
        # Fetch required source data
        try:
            ohlcv_15m_df = await self._registry.get_df(f"{token}_ohlcv_15m_df")
        except KeyError:
            return {h: pd.DataFrame() for h in ("1h", "4h", "8h")}
        # Minimal settings for 15m indicators (kept simple; engine may not need horizon params)
        settings: Dict = {"15m": {"rsi_15m": {"window": 14}, "atr_15m": {"window": 14}}}
        # Offload a single computation to get the wide feature row
        wide_df: pd.DataFrame = await loop.run_in_executor(
            self._pool, self._engine.run_15m_pipeline, token, ohlcv_15m_df, settings
        )
        # Assemble subsets per horizon based on FEATURE_PARAMS keys
        horizons = ["1h", "4h", "8h"]
        results: Dict[str, pd.DataFrame] = {}
        for h in horizons:
            required_cols = list((FP or {}).get(h, {}).keys())
            if not required_cols:
                # If config lacks entries, default to using available columns
                sub = wide_df.copy()
            else:
                existing = [c for c in required_cols if c in wide_df.columns]
                # If none of the required columns exist (e.g., in tests), fall back to the full wide_df
                sub = wide_df.loc[:, existing] if existing else wide_df.copy()
            results[h] = sub
        return results
