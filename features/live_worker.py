from __future__ import annotations

from typing import Dict

import pandas as pd

from core.dataframe_registry import DataFrameRegistry
from features.feature_engine import UnifiedFeatureEngine

__all__ = ["compute_live_features_for_horizon"]


def compute_live_features_for_horizon(
    registry: DataFrameRegistry, horizon: str, tokens: list[str]
) -> pd.DataFrame:
    """
    Top-level picklable function to compute a single-row feature DataFrame for a given
    horizon. Designed to be used with ProcessPoolExecutor.

    This function reconstructs a lightweight UnifiedFeatureEngine using the provided
    registry. It runs the refactored 15m pipeline for the given tokens, and then
    assembles a single-row feature DataFrame for a representative token.
    """
    # Instantiate engine synchronously; underlying operations should remain sync in
    # this process context.
    engine = UnifiedFeatureEngine(registry=registry, sql_engine=None)

    # Minimal settings required by the 15m pipeline feature functions
    settings: Dict = {
        "15m": {
            "rsi_15m": {"window": 14},
            "atr_15m": {"window": 14},
        }
    }

    # Run the async pipeline and then collect latest features into a single row
    import asyncio

    async def _run() -> pd.DataFrame:
        await engine.run_15m_pipeline(tokens=tokens, settings=settings)
        # Choose a representative token deterministically (first in the list)
        token = tokens[0] if tokens else None
        if not token:
            return pd.DataFrame()
        frames = []
        try:
            rsi_df = await registry.get_df(f"{token}_rsi_15m_df")
            frames.append(rsi_df)
        except Exception:
            pass
        try:
            atr_df = await registry.get_df(f"{token}_atr_15m_df")
            frames.append(atr_df)
        except Exception:
            pass
        try:
            atrn_df = await registry.get_df(f"{token}_atr_normalized_15m_df")
            frames.append(atrn_df)
        except Exception:
            pass
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, axis=1)
        try:
            last_row = merged.tail(1).reset_index(drop=True)
        except Exception:
            last_row = merged.iloc[[-1]].reset_index(drop=True)
        return last_row

    result_df = asyncio.run(_run())
    return result_df
