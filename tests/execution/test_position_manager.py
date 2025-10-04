import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from execution.position_manager import PositionManager


class DummyExchange:
    def __init__(self, balance_usd: float = 10000.0, price_map=None):
        self._balance = balance_usd
        self._price_map = price_map or {}

    async def get_portfolio_balance_usd(self) -> float:
        return float(self._balance)

    async def get_current_price(self, symbol: str) -> float:
        return float(self._price_map.get(symbol, 50000.0))


@pytest.mark.asyncio
async def test_position_sizing_percent_risk():
    # Settings configured for percent_risk per spec example; use LEVERAGE=1 to match math
    settings = SimpleNamespace(
        POSITION_SIZING_MODE="percent_risk",
        RISK_PERCENT=1.0,
        MAX_POSITION_SIZE_USD=20000.0,
        LEVERAGE=1.0,
    )
    exchange = DummyExchange(balance_usd=10000.0, price_map={"BTCUSDT": 50000.0})
    risk_manager = SimpleNamespace(is_killswitch_active=lambda: False)

    trade_executor = SimpleNamespace(
        execute_entry=AsyncMock(return_value=True),
        execute_exit=AsyncMock(return_value=True),
    )

    q: asyncio.Queue = asyncio.Queue()
    manager = PositionManager(
        settings=settings,
        risk_manager=risk_manager,
        trade_executor=trade_executor,
        exchange_handlers={"primary": exchange},
        price_tick_queue=q,
        primary_exchange_key="primary",
    )

    size = await manager._calculate_position_size(token="BTC", stop_loss_price=49000.0)
    # Expected: (10000 * 0.01) / (50000 - 49000) = 0.1 BTC
    assert pytest.approx(size, rel=1e-6) == 0.1


@pytest.mark.asyncio
async def test_position_sizing_fixed_usd():
    settings = SimpleNamespace(
        POSITION_SIZING_MODE="usd_fixed",
        FIXED_POSITION_USD=500.0,
        MAX_POSITION_SIZE_USD=20000.0,
        LEVERAGE=5,
    )
    exchange = DummyExchange(balance_usd=10000.0, price_map={"BTCUSDT": 50000.0})
    risk_manager = SimpleNamespace(is_killswitch_active=lambda: False)

    trade_executor = SimpleNamespace(
        execute_entry=AsyncMock(return_value=True),
        execute_exit=AsyncMock(return_value=True),
    )

    q: asyncio.Queue = asyncio.Queue()
    manager = PositionManager(
        settings=settings,
        risk_manager=risk_manager,
        trade_executor=trade_executor,
        exchange_handlers={"primary": exchange},
        price_tick_queue=q,
        primary_exchange_key="primary",
    )

    size = await manager._calculate_position_size(token="BTC", stop_loss_price=49000.0)
    # Expected: 500 / 50000 = 0.01 BTC
    assert pytest.approx(size, rel=1e-6) == 0.01


@pytest.mark.asyncio
async def test_max_position_size_cap():
    settings = SimpleNamespace(
        POSITION_SIZING_MODE="usd_fixed",
        FIXED_POSITION_USD=500.0,
        MAX_POSITION_SIZE_USD=100.0,  # Cap to $100
        LEVERAGE=5,
    )
    exchange = DummyExchange(balance_usd=10000.0, price_map={"BTCUSDT": 50000.0})
    risk_manager = SimpleNamespace(is_killswitch_active=lambda: False)
    trade_executor = SimpleNamespace(
        execute_entry=AsyncMock(return_value=True),
        execute_exit=AsyncMock(return_value=True),
    )

    q: asyncio.Queue = asyncio.Queue()
    manager = PositionManager(
        settings=settings,
        risk_manager=risk_manager,
        trade_executor=trade_executor,
        exchange_handlers={"primary": exchange},
        price_tick_queue=q,
        primary_exchange_key="primary",
    )

    size = await manager._calculate_position_size(token="BTC", stop_loss_price=49000.0)
    # Expected: capped to $100 => 100/50000 = 0.002 BTC
    assert pytest.approx(size, rel=1e-6) == 0.002


@pytest.mark.asyncio
async def test_kill_switch_prevents_trade():
    settings = SimpleNamespace(
        POSITION_SIZING_MODE="usd_fixed",
        FIXED_POSITION_USD=500.0,
        MAX_POSITION_SIZE_USD=20000.0,
        LEVERAGE=5,
    )
    exchange = DummyExchange(balance_usd=10000.0, price_map={"BTCUSDT": 50000.0})
    risk_manager = SimpleNamespace(is_killswitch_active=lambda: True)

    trade_executor = SimpleNamespace(
        execute_entry=AsyncMock(return_value=True),
        execute_exit=AsyncMock(return_value=True),
    )

    q: asyncio.Queue = asyncio.Queue()
    manager = PositionManager(
        settings=settings,
        risk_manager=risk_manager,
        trade_executor=trade_executor,
        exchange_handlers={"primary": exchange},
        price_tick_queue=q,
        primary_exchange_key="primary",
    )

    directive = {
        "directive": "OPEN_LONG",
        "token": "BTC",
        "entry_price": 50000.0,
        "stop_loss": 49000.0,
        "take_profit": 51000.0,
    }

    await manager.process_directive(directive)

    trade_executor.execute_entry.assert_not_called()


@pytest.mark.asyncio
async def test_sl_trigger():
    settings = SimpleNamespace(
        POSITION_SIZING_MODE="usd_fixed",
        FIXED_POSITION_USD=500.0,
        MAX_POSITION_SIZE_USD=20000.0,
        LEVERAGE=5,
    )
    exchange = DummyExchange(balance_usd=10000.0, price_map={"BTCUSDT": 50000.0})
    risk_manager = SimpleNamespace(is_killswitch_active=lambda: False)

    trade_executor = SimpleNamespace(
        execute_entry=AsyncMock(return_value=True),
        execute_exit=AsyncMock(return_value=True),
    )

    q: asyncio.Queue = asyncio.Queue()
    manager = PositionManager(
        settings=settings,
        risk_manager=risk_manager,
        trade_executor=trade_executor,
        exchange_handlers={"primary": exchange},
        price_tick_queue=q,
        primary_exchange_key="primary",
    )

    # Manually add an open long position with SL=49000
    manager._positions["BTCUSDT"] = {
        "token": "BTC",
        "symbol": "BTCUSDT",
        "direction": "OPEN_LONG",
        "entry_price": 50000.0,
        "size": 0.01,
        "stop_loss": 49000.0,
        "take_profit": None,
    }

    # Simulate price tick that breaches SL
    await manager._check_positions_for_triggers({"symbol": "BTCUSDT", "price": 48999.0})

    trade_executor.execute_exit.assert_called_once()
    assert "BTCUSDT" not in manager._positions
