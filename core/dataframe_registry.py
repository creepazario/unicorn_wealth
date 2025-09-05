"""Async-safe registry for in-memory pandas DataFrames.

This module provides a centralized class to store and manage pandas
DataFrames across different asynchronous parts of the application
without race conditions.

Note: This is not a singleton. Create and pass an instance via DI.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List

import pandas as pd

__all__ = ["DataFrameRegistry"]


class DataFrameRegistry:
    """A concurrency-safe registry for pandas DataFrames.

    All access to the internal storage is guarded by an asyncio.Lock to
    prevent race conditions when used from multiple asynchronous tasks.
    """

    def __init__(self) -> None:
        # Internal storage of DataFrames by name
        self._dataframes: Dict[str, pd.DataFrame] = {}
        # Async lock to protect reads/writes to the storage
        self._lock: asyncio.Lock = asyncio.Lock()

    async def get_df(self, name: str) -> pd.DataFrame:
        """Retrieve a stored DataFrame by name.

        Args:
            name: The unique name of the DataFrame.

        Returns:
            The stored pandas DataFrame.

        Raises:
            KeyError: If no DataFrame is stored under the given name.
        """
        async with self._lock:
            if name not in self._dataframes:
                raise KeyError(f"DataFrame '{name}' not found in registry")
            return self._dataframes[name]

    async def store_df(
        self, name: str, df: pd.DataFrame, update_mode: str, storage_period: int
    ) -> None:
        """Store a DataFrame with the specified update behavior.

        Two update modes are supported:
        - "overwrite": replace any existing DataFrame under the name.
        - "rolling_append": append new rows to existing ones and keep only
          the last ``storage_period`` rows.

        Args:
            name: Unique identifier for the DataFrame.
            df: The DataFrame to store.
            update_mode: Either "overwrite" or "rolling_append".
            storage_period: The max number of rows to retain when using
                rolling append mode. Non-positive values result in an
                empty DataFrame being stored.
        """
        # Normalize inputs
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        mode = update_mode.lower().strip()
        keep = int(storage_period)

        async with self._lock:
            if mode == "overwrite":
                self._dataframes[name] = df
                return

            if mode == "rolling_append":
                if name in self._dataframes:
                    combined = pd.concat([self._dataframes[name], df], axis=0)
                else:
                    combined = df

                # Truncate to the last `keep` rows. If keep <= 0 -> empty df
                if keep <= 0:
                    truncated = combined.iloc[0:0]
                else:
                    truncated = combined.tail(keep)

                self._dataframes[name] = truncated
                return

            # Unsupported mode
            raise ValueError(
                "update_mode must be either 'overwrite' or 'rolling_append'"
            )

    async def list_dfs(self) -> List[str]:
        """Return the list of stored DataFrame names."""
        async with self._lock:
            return list(self._dataframes.keys())
