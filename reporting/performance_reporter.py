from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database.models.operational import CurrentPositions, TradeLogs


@dataclass
class _Metrics:
    total_trades: int
    wins: int
    losses: int
    gross_profit: float
    gross_loss: float
    net_pl_usd: float
    avg_pl_pct: float
    win_rate: float
    profit_factor: float


class PerformanceReporter:
    """Aggregates performance across accounts and sends Telegram summaries.

    Dependencies:
      - db_session_factory: async_sessionmaker[AsyncSession]
      - exchange_clients: Dict[str, Any] where each client provides:
          async get_portfolio_balance_usd() -> float
      - notifier: TelegramNotifier-like object with:
          async send_notification(type: str, data: Dict[str, Any], channel_type: str)
    """

    def __init__(
        self,
        db_session_factory: async_sessionmaker[AsyncSession],
        exchange_clients: Dict[str, Any],
        notifier: Any,
    ) -> None:
        self._session_factory = db_session_factory
        self.exchange_clients = exchange_clients or {}
        self.notifier = notifier

    # ---------------- Public API ---------------- #
    async def generate_daily_report(self) -> None:
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=1)
        await self._generate_report(
            since=since,
            title="Daily PnL Summary",
            notif_type="DAILY_PNL_SUMMARY",
        )

    async def generate_weekly_report(self) -> None:
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=7)
        await self._generate_report(
            since=since,
            title="Weekly PnL Summary",
            notif_type="WEEKLY_PNL_SUMMARY",
        )

    # -------------- Internal orchestration -------------- #
    async def _generate_report(
        self, *, since: datetime, title: str, notif_type: str
    ) -> None:
        # Fetch data concurrently: closed trades, open positions, balances
        trades_coro = self._fetch_closed_trades_since(since)
        positions_coro = self._fetch_open_positions()
        balance_coro = self._fetch_total_portfolio_balance_usd()

        trades, positions, total_balance = await asyncio.gather(
            trades_coro, positions_coro, balance_coro
        )

        metrics = self._calculate_metrics(trades)
        report_md = self._format_report(
            title=title,
            as_of=datetime.now(timezone.utc),
            metrics=metrics,
            total_portfolio_usd=total_balance,
            open_positions=positions,
        )

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "message": report_md,
        }
        # Send via Telegram notifier (TRADE channel)
        await self.notifier.send_notification(
            type=notif_type, data=data, channel_type="TRADE"
        )

    # -------------- Data access -------------- #
    async def _fetch_closed_trades_since(self, since_dt: datetime) -> List[TradeLogs]:
        since_ms = int(since_dt.timestamp() * 1000)
        sm = self._session_factory
        async with sm() as session:
            stmt: Select[TradeLogs] = (
                select(TradeLogs)
                .where(TradeLogs.exit_timestamp.is_not(None))
                .where(TradeLogs.exit_timestamp >= since_ms)
                .order_by(TradeLogs.exit_timestamp.asc())
            )
            res = await session.execute(stmt)
            rows: Sequence[TradeLogs] = [r[0] for r in res.all()]
        return list(rows)

    async def _fetch_open_positions(self) -> List[Dict[str, Any]]:
        sm = self._session_factory
        async with sm() as session:
            stmt: Select[CurrentPositions] = select(CurrentPositions)
            res = await session.execute(stmt)
            rows: Sequence[CurrentPositions] = [r[0] for r in res.all()]
        # Convert to simple dicts for reporting
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                out.append(
                    {
                        "account": r.account_name,
                        "token": r.token,
                        "direction": r.direction,
                        "position_size": float(r.position_size),
                        "entry_price": float(r.entry_price),
                        "entry_time": int(r.entry_timestamp),
                        "stop_loss": float(r.virtual_stop_loss),
                        "take_profit": float(r.virtual_take_profit),
                    }
                )
            except Exception:
                # Best-effort; skip malformed rows
                continue
        return out

    async def _fetch_total_portfolio_balance_usd(self) -> float:
        if not self.exchange_clients:
            return 0.0

        async def _one(client: Any) -> float:
            try:
                v = client.get_portfolio_balance_usd()
                if hasattr(v, "__await__"):
                    return float(await v)
                return float(v)
            except Exception:
                return 0.0

        vals = await asyncio.gather(*[_one(c) for c in self.exchange_clients.values()])
        return float(sum(v for v in vals if isinstance(v, (int, float))))

    # -------------- Metrics & formatting -------------- #
    def _calculate_metrics(self, trades: Iterable[TradeLogs]) -> _Metrics:
        total_trades = 0
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        net_pl = 0.0
        pl_pcts: List[float] = []

        for t in trades:
            total_trades += 1
            pnl = float(getattr(t, "pnl_usd", 0.0) or 0.0)
            fees = float(getattr(t, "fees", 0.0) or 0.0)
            # Net P/L = pnl_usd - fees (assuming pnl_usd excludes fees; adjust as needed)
            net = pnl - fees
            net_pl += net
            if net > 0:
                wins += 1
                gross_profit += net
            elif net < 0:
                losses += 1
                gross_loss += abs(net)

            # Percent return based on entry notional (entry_price * position_size)
            try:
                entry_price = float(getattr(t, "entry_price", 0.0) or 0.0)
                size = float(getattr(t, "position_size", 0.0) or 0.0)
                notional = entry_price * size
                if notional > 0:
                    pl_pcts.append((net / notional) * 100.0)
            except Exception:
                pass

        win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        avg_pl_pct = (sum(pl_pcts) / len(pl_pcts)) if pl_pcts else 0.0

        return _Metrics(
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_pl_usd=net_pl,
            avg_pl_pct=avg_pl_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
        )

    def _format_report(
        self,
        *,
        title: str,
        as_of: datetime,
        metrics: _Metrics,
        total_portfolio_usd: float,
        open_positions: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        lines.append(f"📊 {title}")
        lines.append(
            f"• As of: {as_of.strftime('%Y-%m-%d %H:%M:%S %Z') or as_of.isoformat()}"
        )
        lines.append("")
        # KPIs
        pl_str = f"**${metrics.net_pl_usd:,.2f} ({metrics.avg_pl_pct:.2f}%)**"
        lines.append(f"P/L: {pl_str}")
        lines.append(
            f"Trades: {metrics.total_trades} | Win Rate: {metrics.win_rate:.2f}% | Profit Factor: {'∞' if metrics.profit_factor == float('inf') else f'{metrics.profit_factor:.2f}'}"
        )
        lines.append("")
        lines.append(f"Total Portfolio Value: **${float(total_portfolio_usd):,.2f}**")
        lines.append("")
        # Open positions summary
        if open_positions:
            lines.append("Open Positions:")
            for p in open_positions[:20]:  # limit to first 20 for brevity
                try:
                    lines.append(
                        "• {token} {direction} size={size:.4f} entry={entry} SL={sl} TP={tp}".format(
                            token=p.get("token", "?"),
                            direction=p.get("direction", "?"),
                            size=float(p.get("position_size", 0.0) or 0.0),
                            entry=self._fmt_money(p.get("entry_price")),
                            sl=self._fmt_money(p.get("stop_loss")),
                            tp=self._fmt_money(p.get("take_profit")),
                        )
                    )
                except Exception:
                    continue
        else:
            lines.append("Open Positions: None")

        return "\n".join(lines)

    @staticmethod
    def _fmt_money(x: Optional[float]) -> str:
        try:
            if x is None:
                return "?"
            return f"${float(x):,.2f}"
        except Exception:
            return str(x)
