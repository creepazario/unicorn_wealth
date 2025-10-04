from __future__ import annotations

import uuid
import pytest
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

from execution.trade_executor import TradeExecutionEngine
from database.models.operational import TradeLogs


class FakeSession:
    def __init__(self, store: dict[uuid.UUID, TradeLogs]):
        self.store = store
        self.add_called_with: Optional[TradeLogs] = None
        self.commits: int = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, obj: TradeLogs) -> None:
        self.add_called_with = obj
        # simulate add by storing for key trade_id
        if isinstance(obj.trade_id, uuid.UUID):
            self.store[obj.trade_id] = obj

    async def commit(self) -> None:
        self.commits += 1

    async def get(self, model, key):
        if model is TradeLogs:
            return self.store.get(key)
        return None


class FakeSessionMaker:
    def __init__(self, session: FakeSession):
        self._session = session

    def __call__(self):
        # support "async with sm() as session" pattern in executor
        return self._session


@pytest.mark.asyncio
async def test_execute_entry_flow(mocker: Any) -> None:
    # Mock exchange handler
    handler = MagicMock()
    handler.set_leverage = AsyncMock(return_value=None)
    # ccxt-like order response
    order_resp: Dict[str, Any] = {
        "id": "abc123",
        "average": 25000.5,
        "timestamp": 1736200000000,
    }
    handler.place_market_order = AsyncMock(return_value=order_resp)

    # Settings with leverage
    class S:
        LEVERAGE = 7

    settings = S()

    # Fake session infrastructure
    store: dict[uuid.UUID, TradeLogs] = {}
    fake_session = FakeSession(store)
    fake_sm = FakeSessionMaker(fake_session)

    # Mock notifier
    notifier = MagicMock()
    notifier.send_notification = AsyncMock(return_value=None)

    engine_obj = TradeExecutionEngine(
        exchange_handlers={"acct1": handler},
        settings=settings,
        engine_factory=lambda: None,
        notifier=notifier,
    )

    # Patch the executor to return our fake sessionmaker
    mocker.patch.object(engine_obj, "_get_session_maker", return_value=fake_sm)

    result = await engine_obj.execute_entry(
        account="acct1", symbol="BTCUSDT", size=0.1, direction="buy", sl_price=24000.0
    )

    # Assertions on handler calls
    handler.set_leverage.assert_awaited_once_with("BTCUSDT", settings.LEVERAGE)
    handler.place_market_order.assert_awaited_once_with("BTCUSDT", "buy", 0.1)

    # Verify a TradeLogs object was created and added, and commit called
    assert isinstance(fake_session.add_called_with, TradeLogs)
    rec = fake_session.add_called_with
    assert rec is not None
    assert rec.status == "OPEN"
    assert rec.entry_exchange_order_id == order_resp["id"]
    assert rec.entry_price == float(order_resp["average"])
    assert rec.position_size == 0.1
    assert fake_session.commits >= 1

    notifier.send_notification.assert_awaited_once()
    args, kwargs = notifier.send_notification.await_args
    assert args[0] == "ON_POSITION_OPEN"
    assert isinstance(args[1], dict)
    assert args[1]["trade_id"] == result["trade_id"]


@pytest.mark.asyncio
async def test_execute_exit_flow(mocker: Any) -> None:
    # Mock exchange handler
    handler = MagicMock()
    handler.set_leverage = AsyncMock(return_value=None)
    exit_order = {
        "id": "exit789",
        "average": 26000.0,
        "timestamp": 1736201111000,
        "fee": {"cost": 3.5},
    }
    handler.place_market_order = AsyncMock(return_value=exit_order)

    # Settings
    class S:
        LEVERAGE = 3

    settings = S()

    # Fake DB with a seeded TradeLogs entry
    trade_id = uuid.uuid4()
    seed = TradeLogs(
        trade_id=trade_id,
        account_name="acctA",
        exchange="binance",
        token="BTCUSDT",
        direction="buy",
        status="OPEN",
        entry_exchange_order_id="abc",
        entry_price=25000.0,
        position_size=0.1,
        entry_timestamp=1736200000000,
    )
    store: dict[uuid.UUID, TradeLogs] = {trade_id: seed}
    fake_session = FakeSession(store)
    fake_sm = FakeSessionMaker(fake_session)

    notifier = MagicMock()
    notifier.send_notification = AsyncMock(return_value=None)

    engine_obj = TradeExecutionEngine(
        exchange_handlers={"acctA": handler},
        settings=settings,
        engine_factory=lambda: None,
        notifier=notifier,
    )

    mocker.patch.object(engine_obj, "_get_session_maker", return_value=fake_sm)

    # Execute exit
    result = await engine_obj.execute_exit(
        trade_log_id=str(trade_id),
        account="acctA",
        symbol="BTCUSDT",
        size=0.1,
        direction="sell",
    )

    # Verify exchange handler called
    handler.place_market_order.assert_awaited_once_with("BTCUSDT", "sell", 0.1)

    # Verify record updated in fake store
    rec2 = store[trade_id]
    assert rec2.status == "CLOSED"
    assert rec2.exit_exchange_order_id == exit_order["id"]
    assert rec2.exit_price == 26000.0
    assert rec2.exit_timestamp == exit_order["timestamp"]
    assert rec2.fees == 3.5
    # PnL = (26000 - 25000)*0.1 = 100; subtract fees => 96.5
    assert pytest.approx(rec2.pnl_usd, rel=1e-6) == 96.5

    # Notifier called with ON_POSITION_CLOSE and contains pnl
    notifier.send_notification.assert_awaited_once()
    args, kwargs = notifier.send_notification.await_args
    assert args[0] == "ON_POSITION_CLOSE"
    payload = args[1]
    assert isinstance(payload, dict)
    assert payload.get("pnl_usd") is not None
    # Use result to appease linters and validate trade_id echoes input
    assert result["trade_id"] == str(trade_id)
