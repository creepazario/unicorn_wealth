import asyncio
from typing import Dict, List

import pandas as pd
import pytest

from core.live_manager import LiveDataManager


class MockRegistry:
    def __init__(self) -> None:
        self._store: Dict[str, pd.DataFrame] = {}
        self._calls: List[str] = []

    async def store_df(
        self, name: str, df: pd.DataFrame, update_mode: str, storage_period: int
    ) -> None:  # noqa: E501
        # implement overwrite and rolling_append minimal logic
        self._calls.append(name)
        if update_mode == "rolling_append":
            if name in self._store and not self._store[name].empty:
                base = self._store[name]
                # ensure indices compatible
                combined = pd.concat([base, df])
            else:
                combined = df
            if storage_period and storage_period > 0:
                combined = combined.tail(storage_period)
            self._store[name] = combined
        else:
            self._store[name] = df

    async def get_df(self, name: str) -> pd.DataFrame:
        if name not in self._store:
            raise KeyError(name)
        return self._store[name]


@pytest.mark.asyncio
async def test_tick_aggregation_and_resampling():
    # Setup
    registry = MockRegistry()
    queue: asyncio.Queue = asyncio.Queue()
    live = LiveDataManager(
        trade_queue=queue, registry=registry, historical_api_clients=[]
    )

    # Start consumer
    consumer_task = asyncio.create_task(live.run_tick_consumer())

    try:
        # Create synthetic trades for 16 minutes for symbol BTC
        # Start at a fixed aligned minute
        start_ts = pd.Timestamp("2023-01-01T00:00:00Z").value // 10**6  # ms
        # For each minute m, we emit 3 ticks to form OHLCV with predictable values
        # price sequence: open=m+100, then high=m+105, then close=m+102; low=m+99 ensured
        for m in range(16):
            base_ms = start_ts + m * 60_000
            # tick 1 (open)
            await queue.put(
                {
                    "symbol": "BTC",
                    "price": 100 + m,
                    "quantity": 1.0,
                    "timestamp": base_ms + 5_000,
                }
            )
            # tick 2 (high)
            await queue.put(
                {
                    "symbol": "BTC",
                    "price": 105 + m,
                    "quantity": 2.0,
                    "timestamp": base_ms + 20_000,
                }
            )
            # tick 3 (low)
            await queue.put(
                {
                    "symbol": "BTC",
                    "price": 99 + m,
                    "quantity": 4.0,
                    "timestamp": base_ms + 55_000,
                }
            )
            # tick 4 (close last)
            await queue.put(
                {
                    "symbol": "BTC",
                    "price": 102 + m,
                    "quantity": 3.0,
                    "timestamp": base_ms + 59_000,
                }
            )

        # Allow processing
        await asyncio.sleep(0.2)

        # Finalize last open minute by pushing a trade in minute 16 to trigger flush
        # This will finalize minute 15 (0-based) candle
        final_trigger_ms = start_ts + 16 * 60_000 + 1_000
        await queue.put(
            {
                "symbol": "BTC",
                "price": 200.0,
                "quantity": 1.0,
                "timestamp": final_trigger_ms,
            }
        )
        await asyncio.sleep(0.1)

        # Assertions for 1m and 15m
        one_m_key = "BTC_ohlcv_1m_df"
        fifteen_m_key = "BTC_ohlcv_15m_df"

        one_m = await registry.get_df(one_m_key)
        assert isinstance(one_m, pd.DataFrame)
        # We created 16 finalized minutes (0..15)
        assert len(one_m) >= 16
        # Slice the 16 from the end to avoid extra trigger row
        last_16 = one_m.tail(16)
        assert len(last_16) == 16

        # Check first 1m candle (m=0)
        first_row = last_16.iloc[0]
        assert first_row["open"] == 100
        assert first_row["high"] == 105
        assert first_row["low"] == 99
        assert first_row["close"] == 102  # last price before minute end
        # volume sums: 1 + 2 + 3 + 4 = 10
        assert first_row["volume"] == pytest.approx(10.0)

        # 15m aggregation should have 1 completed bar for minutes 0..14
        fifteen = await registry.get_df(fifteen_m_key)
        assert isinstance(fifteen, pd.DataFrame)
        # Expect exactly 1 completed bar
        assert len(fifteen) >= 1
        last15 = fifteen.tail(1)
        row15 = last15.iloc[0]
        # Expected OHLCV across minutes 0..14
        expected_open = 100  # minute 0 open
        expected_high = 105 + 14  # highest high in range
        expected_low = 99  # minute 0 low is the minimum since lows increase by m
        expected_close = 102 + 14  # last closed minute is m=14, close=116
        # Volume: each minute 10 -> 15 minutes = 150
        expected_volume = 10.0 * 15

        assert row15["open"] == expected_open
        assert row15["high"] == expected_high
        assert row15["low"] == expected_low
        assert row15["close"] == expected_close
        assert row15["volume"] == pytest.approx(expected_volume)

    finally:
        consumer_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer_task
