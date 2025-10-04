import asyncio
from typing import List

import pandas as pd
import pytest
from freezegun import freeze_time

from core.live_manager import LiveDataManager


class DummyRegistry:
    async def store_df(
        self, name: str, df: pd.DataFrame, update_mode: str, storage_period: int
    ) -> None:  # noqa: E501
        # no-op for scheduling tests
        return

    async def get_df(self, name: str) -> pd.DataFrame:  # pragma: no cover - unused here
        raise KeyError(name)


async def _recorder(calls: List[str], label: str) -> None:
    calls.append(label)


@pytest.mark.asyncio
async def test_fetch_scheduled_data_at_15_min_mark():
    # 12:15 UTC -> only 15m cadence should run
    queue: asyncio.Queue = asyncio.Queue()
    registry = DummyRegistry()
    live = LiveDataManager(
        trade_queue=queue, registry=registry, historical_api_clients=[]
    )

    # Replace scheduled jobs with recorders
    calls: List[str] = []

    async def job_15m_a():
        await _recorder(calls, "15m_a")

    async def job_1h_a():
        await _recorder(calls, "1h_a")

    async def job_1d_a():
        await _recorder(calls, "1d_a")

    live._scheduled_jobs = {  # type: ignore[attr-defined]
        "15m": [job_15m_a],
        "1h": [job_1h_a],
        "1d": [job_1d_a],
    }

    with freeze_time("2025-09-10 12:15:00", tz_offset=0):
        await live.fetch_scheduled_data()

    assert "15m_a" in calls
    assert "1h_a" not in calls
    assert "1d_a" not in calls


@pytest.mark.asyncio
async def test_fetch_scheduled_data_at_top_of_hour():
    # 13:00 UTC -> 15m and 1h should run
    queue: asyncio.Queue = asyncio.Queue()
    registry = DummyRegistry()
    live = LiveDataManager(
        trade_queue=queue, registry=registry, historical_api_clients=[]
    )

    calls: List[str] = []

    async def job_15m_a():
        await _recorder(calls, "15m_a")

    async def job_1h_a():
        await _recorder(calls, "1h_a")

    async def job_1d_a():
        await _recorder(calls, "1d_a")

    live._scheduled_jobs = {  # type: ignore[attr-defined]
        "15m": [job_15m_a],
        "1h": [job_1h_a],
        "1d": [job_1d_a],
    }

    with freeze_time("2025-09-10 13:00:00", tz_offset=0):
        await live.fetch_scheduled_data()

    assert "15m_a" in calls
    assert "1h_a" in calls
    assert "1d_a" not in calls


@pytest.mark.asyncio
async def test_fetch_scheduled_data_at_midnight():
    # 00:00 UTC -> 15m, 1h, and 1d should run
    queue: asyncio.Queue = asyncio.Queue()
    registry = DummyRegistry()
    live = LiveDataManager(
        trade_queue=queue, registry=registry, historical_api_clients=[]
    )

    calls: List[str] = []

    async def job_15m_a():
        await _recorder(calls, "15m_a")

    async def job_1h_a():
        await _recorder(calls, "1h_a")

    async def job_1d_a():
        await _recorder(calls, "1d_a")

    live._scheduled_jobs = {  # type: ignore[attr-defined]
        "15m": [job_15m_a],
        "1h": [job_1h_a],
        "1d": [job_1d_a],
    }

    with freeze_time("2025-09-10 00:00:00", tz_offset=0):
        await live.fetch_scheduled_data()

    assert "15m_a" in calls
    assert "1h_a" in calls
    assert "1d_a" in calls
