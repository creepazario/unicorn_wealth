from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import pytest

from reporting.performance_reporter import PerformanceReporter


# ---- Lightweight stand-ins for ORM rows ----
@dataclass
class FakeTradeLog:
    entry_price: float
    position_size: float
    pnl_usd: float
    fees: float
    exit_timestamp: int


@dataclass
class FakeCurrentPosition:
    account_name: str
    token: str
    direction: str
    position_size: float
    entry_price: float
    virtual_stop_loss: float
    virtual_take_profit: float
    entry_timestamp: int


# ---- Fake SQLAlchemy bits ----
class FakeSelect:
    def __init__(self, model: Any) -> None:
        self.model = model
        self._where: List[Any] = []
        self._order: Optional[Any] = None

    def where(self, *_):
        # We simply record; evaluation happens in FakeSession
        self._where.extend(_)
        return self

    def order_by(self, *_):
        self._order = _
        return self


class FakeResult:
    def __init__(self, rows: Sequence[Any]):
        self._rows = rows

    def all(self) -> List[Sequence[Any]]:
        # SQLAlchemy returns Row objects; PerformanceReporter indexes [0]
        return [[r] for r in self._rows]


class FakeSession:
    def __init__(
        self, trades: List[FakeTradeLog], positions: List[FakeCurrentPosition]
    ):
        self._trades = trades
        self._positions = positions

    async def execute(self, stmt: Any) -> FakeResult:
        # Determine which model is being selected by checking model attribute on our FakeSelect
        model = getattr(stmt, "_raw_model", None)
        if model is None:
            # The reporter uses sqlalchemy.select(Model), import-free in tests.
            # We can't import those Model classes here; instead we infer by presence of where clauses:
            # If there is any clause that checks exit_timestamp, assume TradeLogs.
            text = str(stmt)
            if "exit_timestamp" in text:
                return FakeResult(self._trades)
            # Otherwise return positions
            return FakeResult(self._positions)
        return FakeResult([])

    async def close(self) -> None:  # pragma: no cover - not used
        pass


class DummyAsyncSessionMaker:
    """A stub implementing async_sessionmaker-like callable returning an async CM.

    The reporter uses: sm = self._session_factory; async with sm() as session:
    So we implement __call__ returning an object that supports async __aenter__/__aexit__.
    """

    def __init__(
        self, trades: List[FakeTradeLog], positions: List[FakeCurrentPosition]
    ):
        self._session = FakeSession(trades, positions)

    def __call__(self):  # returns an async context manager
        return self

    async def __aenter__(self) -> FakeSession:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


# ---- Mocks for exchange handlers and notifier ----
class MockExchange:
    def __init__(self, balance: float) -> None:
        self._balance = balance

    async def get_portfolio_balance_usd(self) -> float:
        await asyncio.sleep(0)
        return float(self._balance)


class SpyNotifier:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def send_notification(
        self, type: str, data: Dict[str, Any], channel_type: str
    ) -> None:
        self.calls.append({"type": type, "data": data, "channel_type": channel_type})


# ---- Helpers for building timestamps ----
def now_ms(offset_days: int = 0) -> int:
    dt = datetime.now(timezone.utc) + timedelta(days=offset_days)
    return int(dt.timestamp() * 1000)


@pytest.mark.asyncio
async def test_generate_daily_report_calculations():
    # Build trade set within last 24h (mix of win/loss)
    # Trade A: entry 100, size 1, pnl 10, fees 1 -> net 9 (win)
    # percent = 9/100 = 9%
    t1 = FakeTradeLog(
        entry_price=100.0,
        position_size=1.0,
        pnl_usd=10.0,
        fees=1.0,
        exit_timestamp=now_ms(-0),
    )
    # Trade B: entry 200, size 0.5, pnl -15, fees 0.5 -> net -15.5 (loss)
    # percent = -15.5/100 = -15.5%
    t2 = FakeTradeLog(
        entry_price=200.0,
        position_size=0.5,
        pnl_usd=-15.0,
        fees=0.5,
        exit_timestamp=now_ms(-0),
    )
    # Trade C: entry 50, size 2, pnl 8, fees 0 -> net 8 (win)
    # percent = 8/100 = 8%
    t3 = FakeTradeLog(
        entry_price=50.0,
        position_size=2.0,
        pnl_usd=8.0,
        fees=0.0,
        exit_timestamp=now_ms(-0),
    )

    positions = [
        FakeCurrentPosition(
            account_name="acc1",
            token="BTC",
            direction="OPEN_LONG",
            position_size=0.1234,
            entry_price=30000.0,
            virtual_stop_loss=29000.0,
            virtual_take_profit=33000.0,
            entry_timestamp=now_ms(-1),
        ),
        FakeCurrentPosition(
            account_name="acc2",
            token="ETH",
            direction="OPEN_SHORT",
            position_size=1.5,
            entry_price=1800.0,
            virtual_stop_loss=1850.0,
            virtual_take_profit=1700.0,
            entry_timestamp=now_ms(-2),
        ),
    ]

    session_maker = DummyAsyncSessionMaker(trades=[t1, t2, t3], positions=positions)

    exchanges = {"ex1": MockExchange(10000.0), "ex2": MockExchange(5000.0)}
    notifier = SpyNotifier()

    reporter = PerformanceReporter(session_maker, exchanges, notifier)

    await reporter.generate_daily_report()

    # Verify notifier call
    assert notifier.calls, "Notifier was not called"
    call = notifier.calls[-1]
    assert call["type"] == "DAILY_PNL_SUMMARY"
    assert call["channel_type"] == "TRADE"
    message = call["data"]["message"]

    # Expected KPI calculations
    total_trades = 3
    wins = 2
    net_pl = (10.0 - 1.0) + (-15.0 - 0.5) + (8.0 - 0.0)
    # per-trade %: 9%, -15.5%, 8% -> avg = (9 - 15.5 + 8)/3 = 0.5%
    avg_pct = (9.0 - 15.5 + 8.0) / 3.0
    win_rate = wins / total_trades * 100.0
    gross_profit = 9.0 + 8.0
    gross_loss = 15.5
    profit_factor = gross_profit / gross_loss

    # Assert message contains key metrics
    assert f"P/L: **${net_pl:,.2f} ({avg_pct:.2f}%)**" in message
    assert f"Trades: {total_trades} | Win Rate: {win_rate:.2f}%" in message
    assert f"Profit Factor: {profit_factor:.2f}" in message

    # Portfolio total
    total_portfolio = 15000.0
    assert f"Total Portfolio Value: **${total_portfolio:,.2f}**" in message

    # Open positions summary snippets
    assert "Open Positions:" in message
    assert "• BTC OPEN_LONG" in message
    assert "• ETH OPEN_SHORT" in message


@pytest.mark.asyncio
async def test_generate_weekly_report():
    # Create one trade within 7 days and one older than 7 days
    recent_trade = FakeTradeLog(
        entry_price=100.0,
        position_size=1.0,
        pnl_usd=20.0,
        fees=2.0,
        exit_timestamp=now_ms(-1),  # within 7 days
    )
    # Our simple FakeSession doesn't filter by timestamp; to emulate weekly behavior, only pass recent trades
    session_maker = DummyAsyncSessionMaker(trades=[recent_trade], positions=[])

    exchanges = {"ex1": MockExchange(7777.0)}
    notifier = SpyNotifier()

    reporter = PerformanceReporter(session_maker, exchanges, notifier)
    await reporter.generate_weekly_report()

    assert notifier.calls, "Notifier was not called for weekly report"
    call = notifier.calls[-1]
    assert call["type"] == "WEEKLY_PNL_SUMMARY"
    assert call["channel_type"] == "TRADE"
    message = call["data"]["message"]

    net_pl = 18.0  # 20 - 2
    avg_pct = 18.0 / 100.0 * 100.0  # 18%
    assert f"P/L: **${net_pl:,.2f} ({avg_pct:.2f}%)**" in message
    assert "Total Portfolio Value: **$7,777.00**" in message
    assert "Open Positions: None" in message
