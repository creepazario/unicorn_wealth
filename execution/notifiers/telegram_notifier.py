from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    # Telethon is optional at import-time to keep modules importable in test envs
    from telethon import TelegramClient  # type: ignore
except Exception:  # pragma: no cover - runtime validation in start()
    TelegramClient = None  # type: ignore


class TelegramNotifier:
    """Asynchronous notifier for sending system messages to Telegram.

    This class centralizes all Telegram notifications. It is configurable via a
    Settings object. Notification sending is skipped silently when disabled at
    the master switch level or per-notification type.

    Settings attributes consumed (case-sensitive, snake_case):
      - telegram_api_id: str
      - telegram_api_hash: str
      - telegram_bot_token: str
      - telegram_admin_channel_id: str
      - telegram_trade_channel_id: str
      - TELEGRAM_ENABLED (from config module) via getattr fallback
      - NOTIFICATION_SETTINGS (from config module) via getattr fallback

    Note: The project mixes lowercase attributes in Settings with UPPER_CASE
    values sourced from config.py. For backward compatibility we check both
    the Settings instance and the config module via getattr fallbacks.
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        # Always defer client construction to start() to allow tests to patch
        self.client = None
        self._started = False

    async def start(self) -> None:
        """Initialize and connect the Telethon client using bot token.

        Should be called once at application startup.
        """
        if self._started:
            return
        if TelegramClient is None:
            raise RuntimeError(
                "Telethon is not installed. Please install telethon to use TelegramNotifier."
            )
        if self.client is None:
            # Construct client; in tests, TelegramClient is patched before this call
            self.client = TelegramClient(
                "unicorn_wealth_session",
                getattr(self.settings, "telegram_api_id"),
                getattr(self.settings, "telegram_api_hash"),
            )
        await self.client.start(bot_token=getattr(self.settings, "telegram_bot_token"))
        self._started = True

    async def send_notification(
        self, type: str, data: Dict[str, Any], channel_type: str
    ) -> None:
        """Send a notification if enabled.

        Args:
            type: Logical notification type, e.g., 'ON_POSITION_OPEN', 'ON_POSITION_CLOSE', 'SYSTEM_ALERTS'.
            data: Payload used by formatting functions.
            channel_type: 'ADMIN' or 'TRADE' to map to the proper channel.
        """
        # Master switch
        telegram_enabled = self._get_setting("TELEGRAM_ENABLED", default=True)
        if not telegram_enabled:
            return

        # Per-type switch
        notification_settings = self._get_setting("NOTIFICATION_SETTINGS", default={})
        if isinstance(notification_settings, dict):
            if not notification_settings.get(type, True):
                return

        # Ensure client started
        if not self._started:
            await self.start()

        # Channel mapping
        channel_id = self._resolve_channel(channel_type)
        if channel_id is None:
            # Silently ignore if channel cannot be resolved
            return

        # Format message based on type
        formatter = {
            "ON_POSITION_OPEN": self._format_position_opened,
            "ON_POSITION_CLOSE": self._format_position_closed,
            "SYSTEM_ALERTS": self._format_system_alert,
        }.get(type, self._format_system_alert)

        message = formatter(data or {})
        if not message:
            return

        # Send
        await self.client.send_message(
            entity=channel_id, message=message, parse_mode="md"
        )

    # -----------------------
    # Private helpers
    # -----------------------
    def _get_setting(self, name: str, default: Any) -> Any:
        """Retrieve setting from Settings object or module-level config fallback.

        This supports both uppercase constants defined in config.py and
        lowercase fields in the Settings Pydantic model.
        """
        # Direct attribute on Settings instance
        if hasattr(self.settings, name):
            return getattr(self.settings, name)
        # Lowercase variant for pydantic fields
        lc = name.lower()
        if hasattr(self.settings, lc):
            return getattr(self.settings, lc)
        # Fallback to config module if imported
        try:
            import config as module_config  # local project module

            if hasattr(module_config, name):
                return getattr(module_config, name)
        except Exception:
            pass
        return default

    def _resolve_channel(self, channel_type: str) -> Optional[str]:
        key = (channel_type or "").strip().upper()
        if key == "ADMIN":
            return getattr(self.settings, "telegram_admin_channel_id", None)
        if key == "TRADE":
            return getattr(self.settings, "telegram_trade_channel_id", None)
        return None

    def _format_position_opened(self, data: Dict[str, Any]) -> str:
        ts = self._fmt_ts(data.get("timestamp"))
        exchange = data.get("exchange", "?")
        account = data.get("account", "?")
        token = data.get("token", data.get("symbol", "?"))
        direction = data.get("direction", "?")
        entry_price = data.get("entry_price")
        size = data.get("position_size")
        return (
            f"🟢 Position Opened\n"
            f"• Time: {ts}\n"
            f"• Exchange: {exchange}\n"
            f"• Account: {account}\n"
            f"• Token: {token}\n"
            f"• Direction: {direction}\n"
            f"• Entry: {self._fmt_price(entry_price)}\n"
            f"• Size: {self._fmt_size(size)}"
        )

    def _format_position_closed(self, data: Dict[str, Any]) -> str:
        ts = self._fmt_ts(data.get("timestamp"))
        exchange = data.get("exchange", "?")
        account = data.get("account", "?")
        token = data.get("token", data.get("symbol", "?"))
        direction = data.get("direction", "?")
        entry_price = data.get("entry_price")
        exit_price = data.get("exit_price")
        size = data.get("position_size")
        pl_usd = data.get("pl_usd")
        pl_pct = data.get("pl_pct")
        fees = data.get("fees")
        duration = data.get("duration")
        pl_str = (
            f"**{self._fmt_money(pl_usd)} ({self._fmt_pct(pl_pct)})**"
            if pl_usd is not None
            else "?"
        )
        return (
            f"🔴 Position Closed\n"
            f"• Time: {ts}\n"
            f"• Exchange: {exchange}\n"
            f"• Account: {account}\n"
            f"• Token: {token}\n"
            f"• Direction: {direction}\n"
            f"• Entry: {self._fmt_price(entry_price)}\n"
            f"• Exit: {self._fmt_price(exit_price)}\n"
            f"• Size: {self._fmt_size(size)}\n"
            f"• P/L: {pl_str}\n"
            f"• Fees: {self._fmt_money(fees)}\n"
            f"• Duration: {self._fmt_duration(duration)}"
        )

    def _format_system_alert(self, data: Dict[str, Any]) -> str:
        ts = self._fmt_ts(data.get("timestamp"))
        title = data.get("title", "System Alert")
        message = data.get("message", "")
        severity = data.get("severity", "INFO")
        return f"⚠️ {title}\n" f"• Time: {ts}\n" f"• Severity: {severity}\n" f"{message}"

    # ---- Formatting utilities ----
    @staticmethod
    def _fmt_ts(ts: Optional[Any]) -> str:
        if ts is None:
            ts_dt = datetime.now(timezone.utc)
        elif isinstance(ts, (int, float)):
            ts_dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        elif isinstance(ts, str):
            try:
                ts_dt = datetime.fromisoformat(ts)
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            except Exception:
                ts_dt = datetime.now(timezone.utc)
        elif isinstance(ts, datetime):
            ts_dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        else:
            ts_dt = datetime.now(timezone.utc)
        return ts_dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    @staticmethod
    def _fmt_price(x: Optional[float]) -> str:
        if x is None:
            return "?"
        try:
            return f"${float(x):,.2f}"
        except Exception:
            return str(x)

    @staticmethod
    def _fmt_money(x: Optional[float]) -> str:
        if x is None:
            return "?"
        try:
            return f"${float(x):,.2f}"
        except Exception:
            return str(x)

    @staticmethod
    def _fmt_size(x: Optional[float]) -> str:
        if x is None:
            return "?"
        try:
            return f"{float(x):,.4f}"
        except Exception:
            return str(x)

    @staticmethod
    def _fmt_pct(x: Optional[float]) -> str:
        if x is None:
            return "?"
        try:
            return f"{float(x):.2f}%"
        except Exception:
            return str(x)

    @staticmethod
    def _fmt_duration(x: Optional[Any]) -> str:
        if x is None:
            return "?"
        if isinstance(x, (int, float)):
            # assume seconds
            seconds = int(x)
        elif isinstance(x, str):
            try:
                seconds = int(float(x))
            except Exception:
                return x
        else:
            return str(x)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        parts = []
        if d:
            parts.append(f"{d}d")
        if h:
            parts.append(f"{h}h")
        if m:
            parts.append(f"{m}m")
        if s or not parts:
            parts.append(f"{s}s")
        return " ".join(parts)
