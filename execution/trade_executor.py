from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, Mapping, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.config_loader import Settings
from database.models.operational import TradeLogs


class TradeExecutionEngine:
    """Executes trades via exchange handlers, enforces leverage, and logs to DB.

    This engine receives high-level commands (e.g., from a PositionManager) and
    interacts with exchange handlers (wrapping ccxt) to place orders. It writes a
    permanent audit trail into TradeLogs and emits notifier events.
    """

    def __init__(
        self,
        *,
        exchange_handlers: Mapping[str, Any],
        settings: Settings | Any,
        engine_factory: Callable[[], AsyncEngine],
        notifier: Any,
    ) -> None:
        self.handlers = dict(exchange_handlers)
        self.settings = settings
        self._engine_factory = engine_factory
        self.notifier = notifier

        # Lazily-built sessionmaker to avoid creating engines eagerly
        self._session_maker: Optional[async_sessionmaker[AsyncSession]] = None

    def _get_session_maker(self) -> async_sessionmaker[AsyncSession]:
        if self._session_maker is None:
            engine = self._engine_factory()
            self._session_maker = async_sessionmaker(engine, expire_on_commit=False)
        return self._session_maker

    async def execute_entry(
        self,
        account: str,
        symbol: str,
        size: float,
        direction: str,
        sl_price: float,
    ) -> Dict[str, Any]:
        """Execute an entry (market) order and log it.

        Returns a dict containing at least trade_id and basic order details.
        """
        handler = self._select_handler(account)

        leverage = int(getattr(self.settings, "LEVERAGE", 1))
        await handler.set_leverage(symbol, leverage)

        order = await handler.place_market_order(symbol, direction, size)

        entry_price = _extract_price(order)
        entry_order_id = str(order.get("id")) if order is not None else None
        entry_ts = _extract_timestamp_ms(order)

        trade_id = uuid.uuid4()
        exchange_name = handler.__class__.__name__.replace("Client", "").lower()

        # Persist log
        sm = self._get_session_maker()
        async with sm() as session:
            await self._create_trade_log(
                session=session,
                trade_id=trade_id,
                account=account,
                exchange=exchange_name,
                symbol=symbol,
                direction=direction,
                status="OPEN",
                entry_price=entry_price,
                size=size,
                entry_timestamp=entry_ts,
                entry_exchange_order_id=entry_order_id,
            )

        payload = {
            "trade_id": str(trade_id),
            "account": account,
            "exchange": exchange_name,
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "entry_price": entry_price,
            "entry_timestamp": entry_ts,
            "sl_price": sl_price,
            "order": order,
        }
        # Fire-and-forget notifier if provided
        if hasattr(self.notifier, "send_notification"):
            await maybe_await(
                self.notifier.send_notification("ON_POSITION_OPEN", payload)
            )

        return payload

    async def execute_exit(
        self,
        trade_log_id: str,
        account: str,
        symbol: str,
        size: float,
        direction: str,
    ) -> Dict[str, Any]:
        """Execute an exit (market) order, update DB, and notify."""
        handler = self._select_handler(account)
        order = await handler.place_market_order(symbol, direction, size)

        exit_price = _extract_price(order)
        exit_order_id = str(order.get("id")) if order is not None else None
        exit_ts = _extract_timestamp_ms(order)
        fees = _extract_fees(order)

        # Update log
        sm = self._get_session_maker()
        full_record: Optional[Dict[str, Any]] = None
        async with sm() as session:
            full_record = await self._update_trade_log_on_exit(
                session=session,
                trade_id=trade_log_id,
                exit_price=exit_price,
                exit_timestamp=exit_ts,
                fees=fees,
                exit_exchange_order_id=exit_order_id,
            )

        if full_record is None:
            full_record = {  # fallback minimal payload
                "trade_id": trade_log_id,
                "account": account,
                "symbol": symbol,
                "direction": direction,
                "size": size,
                "exit_price": exit_price,
                "exit_timestamp": exit_ts,
                "fees": fees,
                "order": order,
            }

        if hasattr(self.notifier, "send_notification"):
            await maybe_await(
                self.notifier.send_notification("ON_POSITION_CLOSE", full_record)
            )

        return full_record

    def _select_handler(self, account: str) -> Any:
        try:
            return self.handlers[account]
        except KeyError as exc:
            raise KeyError(
                f"No exchange handler configured for account '{account}'"
            ) from exc

    async def _create_trade_log(
        self,
        *,
        session: AsyncSession,
        trade_id: uuid.UUID,
        account: str,
        exchange: str,
        symbol: str,
        direction: str,
        status: str,
        entry_price: float,
        size: float,
        entry_timestamp: int,
        entry_exchange_order_id: Optional[str],
    ) -> None:
        rec = TradeLogs(
            trade_id=trade_id,
            account_name=account,
            exchange=exchange,
            token=symbol,
            direction=direction,
            status=status,
            entry_exchange_order_id=entry_exchange_order_id,
            entry_price=float(entry_price),
            position_size=float(size),
            entry_timestamp=int(entry_timestamp),
        )
        session.add(rec)
        await session.commit()

    async def _update_trade_log_on_exit(
        self,
        *,
        session: AsyncSession,
        trade_id: str,
        exit_price: Optional[float],
        exit_timestamp: int,
        fees: Optional[float],
        exit_exchange_order_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        # Load row
        rec: Optional[TradeLogs] = await session.get(TradeLogs, uuid.UUID(trade_id))
        if rec is None:
            return None

        rec.exit_price = float(exit_price) if exit_price is not None else None
        rec.exit_timestamp = int(exit_timestamp)
        rec.fees = float(fees) if fees is not None else None
        rec.status = "CLOSED"
        rec.exit_exchange_order_id = exit_exchange_order_id

        # Compute PnL if both prices available
        if rec.entry_price is not None and rec.exit_price is not None:
            pnl = (rec.exit_price - rec.entry_price) * (rec.position_size or 0.0)
            # If short, invert
            if isinstance(rec.direction, str) and rec.direction.lower() in {
                "sell",
                "short",
            }:
                pnl = -pnl
            # Subtract fees if present
            if rec.fees:
                pnl -= rec.fees
            rec.pnl_usd = pnl

        await session.commit()

        # Build dict view for notifier
        return {
            "trade_id": str(rec.trade_id),
            "account": rec.account_name,
            "exchange": rec.exchange,
            "symbol": rec.token,
            "direction": rec.direction,
            "size": rec.position_size,
            "entry_price": rec.entry_price,
            "entry_timestamp": rec.entry_timestamp,
            "exit_price": rec.exit_price,
            "exit_timestamp": rec.exit_timestamp,
            "fees": rec.fees,
            "pnl_usd": rec.pnl_usd,
            "status": rec.status,
            "entry_exchange_order_id": rec.entry_exchange_order_id,
            "exit_exchange_order_id": rec.exit_exchange_order_id,
        }


def _extract_price(order: Optional[Dict[str, Any]]) -> Optional[float]:
    if not order:
        return None
    price = (
        order.get("average")
        or order.get("price")
        or order.get("info", {}).get("avgPrice")
    )
    try:
        return float(price) if price is not None else None
    except Exception:
        return None


def _extract_timestamp_ms(order: Optional[Dict[str, Any]]) -> int:
    if not order:
        return int(time.time() * 1000)
    ts = order.get("timestamp") or order.get("lastTradeTimestamp")
    try:
        if ts is None:
            return int(time.time() * 1000)
        return int(ts)
    except Exception:
        return int(time.time() * 1000)


def _extract_fees(order: Optional[Dict[str, Any]]) -> Optional[float]:
    if not order:
        return None
    # ccxt orders may have a fee dict or a list under 'fees'
    fee = order.get("fee")
    if isinstance(fee, dict):
        cost = fee.get("cost")
        try:
            return float(cost) if cost is not None else None
        except Exception:
            return None
    fees = order.get("fees")
    if isinstance(fees, list) and fees:
        try:
            return (
                float(fees[0].get("cost")) if fees[0].get("cost") is not None else None
            )
        except Exception:
            return None
    return None


async def maybe_await(result: Any) -> Any:
    if callable(getattr(result, "__await__", None)):
        return await result  # type: ignore[misc]
    return result
