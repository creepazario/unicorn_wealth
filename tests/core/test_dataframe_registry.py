import asyncio
from typing import List

import pandas as pd
import pytest

from unicorn_wealth.core.dataframe_registry import DataFrameRegistry


@pytest.fixture()
def registry() -> DataFrameRegistry:
    return DataFrameRegistry()


@pytest.mark.asyncio
async def test_store_and_get_df(registry: DataFrameRegistry) -> None:
    # Prepare a simple DataFrame
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    # Store with overwrite and then retrieve
    await registry.store_df("test", df1, update_mode="overwrite", storage_period=10)
    got = await registry.get_df("test")

    # Validate content equality
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), df1.reset_index(drop=True)
    )

    # Overwrite with a new dataframe
    df2 = pd.DataFrame({"a": [9], "b": ["q"]})
    await registry.store_df("test", df2, update_mode="overwrite", storage_period=10)
    got2 = await registry.get_df("test")
    pd.testing.assert_frame_equal(
        got2.reset_index(drop=True), df2.reset_index(drop=True)
    )

    # list_dfs should contain the stored key
    names: List[str] = await registry.list_dfs()
    assert "test" in names


@pytest.mark.asyncio
async def test_rolling_append(registry: DataFrameRegistry) -> None:
    name = "rolling"
    period = 5

    # First append 3 rows
    df_a = pd.DataFrame({"x": [1, 2, 3]})
    await registry.store_df(
        name, df_a, update_mode="rolling_append", storage_period=period
    )

    # Then append 4 rows -> total would be 7, but truncated to last 5
    df_b = pd.DataFrame({"x": [4, 5, 6, 7]})
    await registry.store_df(
        name, df_b, update_mode="rolling_append", storage_period=period
    )

    final = await registry.get_df(name)
    assert len(final) == 5
    # Expect to keep last 5 values: 3,4,5,6,7
    pd.testing.assert_frame_equal(
        final.reset_index(drop=True), pd.DataFrame({"x": [3, 4, 5, 6, 7]})
    )


@pytest.mark.asyncio
async def test_concurrent_access_is_safe(registry: DataFrameRegistry) -> None:
    name = "concurrent"
    tasks = 4
    iterations = 200
    storage_period = 10_000  # large enough to avoid truncation

    async def writer(task_id: int) -> None:
        for i in range(iterations):
            # Each write appends a single row unique to this task/iteration
            df = pd.DataFrame({"task": [task_id], "i": [i]})
            await registry.store_df(
                name, df, update_mode="rolling_append", storage_period=storage_period
            )
            # Yield control to increase interleaving
            await asyncio.sleep(0)

    await asyncio.gather(*(writer(t) for t in range(tasks)))

    # After concurrent writes, the final length should equal total rows appended
    final = await registry.get_df(name)
    assert len(final) == tasks * iterations

    # Optional sanity checks: ensure no NaNs introduced and columns are intact
    assert set(final.columns) == {"task", "i"}
    assert final.isna().sum().sum() == 0
