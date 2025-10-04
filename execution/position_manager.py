from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional


class PositionManager:
    """Tracks open positions, sizes new trades, and manages exits.

    This class bridges strategic directives from ConsensusEngine to the
    TradeExecutionEngine. It maintains an in-memory map of open positions and
    monitors price ticks to trigger virtual stop-loss and take-profit exits.

    Dependencies are injected to keep this component testable and decoupled from
    concrete exchange implementations.
    """

    def __init__(
        self,
        settings: Any,
        risk_manager: Any,
        trade_executor: Any,
        exchange_handlers: Dict[str, Any],
        price_tick_queue: asyncio.Queue,
        primary_exchange_key: str | None = None,
    ) -> None:
        """Initialize the PositionManager.

        Args:
            settings: Settings module (e.g., config.py) with required attributes.
            risk_manager: RiskManager instance (must have is_killswitch_active()).
            trade_executor: TradeExecutionEngine-like object with execute_entry/exit.
            exchange_handlers: Dict of exchange clients; primary one used for
                balance and price lookup. Each handler should provide:
                  - async get_portfolio_balance_usd() -> float
                  - async get_current_price(symbol: str) -> float
            price_tick_queue: asyncio.Queue receiving live price ticks with fields
                {'symbol': 'BTCUSDT', 'price': float} or {'token': 'BTC', 'price': ...}.
            primary_exchange_key: Optional key to choose primary exchange from
                exchange_handlers. If None, first dict key is used.
        """
        self.settings = settings
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor
        self.exchange_handlers = exchange_handlers
        self.price_tick_queue: asyncio.Queue = price_tick_queue
        # Pick a primary exchange for balance and price queries
        if primary_exchange_key and primary_exchange_key in exchange_handlers:
            self.primary_exchange_key = primary_exchange_key
        else:
            self.primary_exchange_key = (
                next(iter(exchange_handlers)) if exchange_handlers else ""
            )
        self._log = logging.getLogger(__name__)

        # Map: symbol -> position dict
        self._positions: Dict[str, Dict[str, Any]] = {}

    async def run_price_monitor(self) -> None:
        """Consume price ticks and evaluate exit triggers indefinitely."""
        while True:
            tick = await self.price_tick_queue.get()
            try:
                await self._check_positions_for_triggers(tick)
            finally:
                # Mark task done to support Queue.join() if used
                self.price_tick_queue.task_done()

    async def process_directive(self, directive: Dict[str, Any]) -> None:
        """Process a directive from ConsensusEngine.

        Only OPEN_LONG / OPEN_SHORT directives result in new entries here. Exit
        logic is handled by price triggers or future opposing signals.
        """
        action = directive.get("directive")
        token = directive.get("token") or directive.get("symbol")
        if not action or not token:
            return

        if action not in ("OPEN_LONG", "OPEN_SHORT"):
            # Ignore non-open directives for now
            return

        if getattr(self.risk_manager, "is_killswitch_active", lambda: False)():
            self._log.warning("Kill-switch active; aborting new position for %s", token)
            return

        stop_loss = directive.get("stop_loss")
        take_profit = directive.get("take_profit")
        entry_price = directive.get("entry_price")

        # Determine symbol formatting; default to e.g., BTCUSDT
        quote = "USDT"
        symbol = directive.get("symbol") or f"{token}{quote}"

        try:
            size_qty = await self._calculate_position_size(
                token=token,
                stop_loss_price=float(stop_loss) if stop_loss is not None else None,
            )
        except Exception as exc:
            self._log.exception("Failed sizing for %s: %s", token, exc)
            return

        if size_qty is None or size_qty <= 0:
            self._log.info("Calculated size is non-positive; skipping %s", token)
            return

        # Execute entry via trade executor; be tolerant to interface specifics
        try:
            result = await self._maybe_await(
                self.trade_executor.execute_entry(
                    symbol=symbol,
                    side="BUY" if action == "OPEN_LONG" else "SELL",
                    size=size_qty,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=getattr(self.settings, "LEVERAGE", 1),
                )
            )
        except Exception as exc:
            self._log.exception("Entry execution failed for %s: %s", symbol, exc)
            return

        if result is False:
            # Execution component may return False on failure
            self._log.error("Entry execution returned failure for %s", symbol)
            return

        # Update in-memory state
        self._positions[symbol] = {
            "token": token,
            "symbol": symbol,
            "direction": action,
            "entry_price": entry_price,
            "size": size_qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        self._log.info("Opened position %s: %s", symbol, self._positions[symbol])

    async def _check_positions_for_triggers(self, price_tick: Dict[str, Any]) -> None:
        """Check live price against SL/TP for the relevant position.

        price_tick should include either 'symbol' or 'token' and a 'price'.
        """
        price = price_tick.get("price")
        if price is None:
            return
        symbol = price_tick.get("symbol")
        if not symbol:
            token = price_tick.get("token") or price_tick.get("asset")
            if not token:
                return
            symbol = f"{token}USDT"

        pos = self._positions.get(symbol)
        if not pos:
            return

        sl = pos.get("stop_loss")
        tp = pos.get("take_profit")
        direction = pos.get("direction")

        breached = False
        exit_reason: Optional[str] = None
        if direction == "OPEN_LONG":
            if sl is not None and price <= float(sl):
                breached = True
                exit_reason = "STOP_LOSS"
            elif tp is not None and price >= float(tp):
                breached = True
                exit_reason = "TAKE_PROFIT"
        elif direction == "OPEN_SHORT":
            if sl is not None and price >= float(sl):
                breached = True
                exit_reason = "STOP_LOSS"
            elif tp is not None and price <= float(tp):
                breached = True
                exit_reason = "TAKE_PROFIT"

        if breached:
            try:
                await self._maybe_await(
                    self.trade_executor.execute_exit(
                        symbol=symbol,
                        reason=exit_reason,
                        size=pos.get("size"),
                    )
                )
            except Exception as exc:
                self._log.exception("Exit execution failed for %s: %s", symbol, exc)
                return
            # Remove position on successful trigger attempt regardless of result to
            # avoid repeated attempts; in real system we might check result
            self._positions.pop(symbol, None)
            self._log.info(
                "Closed position %s due to %s at price=%s", symbol, exit_reason, price
            )

    async def _calculate_position_size(
        self, token: str, stop_loss_price: Optional[float]
    ) -> Optional[float]:
        """Calculate asset quantity based on configured sizing mode.

        Returns the asset quantity (e.g., BTC size). Uses current price from the
        primary exchange. Applies MAX_POSITION_SIZE_USD cap.
        """
        if not self.exchange_handlers:
            self._log.error("No exchange handlers configured; cannot size position")
            return None
        primary = self.exchange_handlers[self.primary_exchange_key]

        # Acquire balances and price
        balance_usd = float(
            await self._maybe_await(primary.get_portfolio_balance_usd())
        )
        current_price = float(
            await self._maybe_await(primary.get_current_price(f"{token}USDT"))
        )
        if current_price <= 0:
            self._log.error("Invalid current price for %s: %s", token, current_price)
            return None

        mode = getattr(self.settings, "POSITION_SIZING_MODE", "percent_risk")
        max_pos_usd = float(getattr(self.settings, "MAX_POSITION_SIZE_USD", 0.0) or 0.0)
        leverage = float(getattr(self.settings, "LEVERAGE", 1) or 1)

        # Calculate notional USD exposure prior to leverage (i.e., position value)
        notional_usd: float
        if mode == "percent_risk":
            risk_pct = float(getattr(self.settings, "RISK_PERCENT", 1.0)) / 100.0
            if stop_loss_price is None:
                # Without SL distance we cannot compute risk-based position; fallback to zero
                self._log.warning(
                    "percent_risk mode requires stop_loss_price; returning 0 size"
                )
                return 0.0
            # Risk per trade in USD
            risk_usd = balance_usd * risk_pct
            # Price distance to SL
            distance = abs(current_price - float(stop_loss_price))
            if distance <= 0:
                self._log.warning(
                    "Stop-loss distance is non-positive; returning 0 size"
                )
                return 0.0
            # For leveraged positions, risk per USD notional is (distance/current_price)/leverage
            # Position notional = risk_usd / (distance/current_price/leverage)
            notional_usd = risk_usd * current_price * leverage / distance
        elif mode == "percent_fixed":
            pct = float(getattr(self.settings, "FIXED_POSITION_PERCENT", 0.0)) / 100.0
            notional_usd = balance_usd * pct
        elif mode == "usd_fixed":
            notional_usd = float(getattr(self.settings, "FIXED_POSITION_USD", 0.0))
        else:
            self._log.warning("Unknown POSITION_SIZING_MODE=%s; returning 0 size", mode)
            return 0.0

        # Apply global cap
        if max_pos_usd > 0:
            notional_usd = min(notional_usd, max_pos_usd)

        if notional_usd <= 0:
            return 0.0

        # Convert notional USD to asset quantity (qty = notional / price)
        qty = notional_usd / current_price
        return float(qty)

    async def _maybe_await(self, v: Any) -> Any:
        if hasattr(v, "__await__"):
            return await v
        return v
