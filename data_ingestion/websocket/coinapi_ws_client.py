from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from core.config_loader import Settings

try:  # runtime dependency
    import websockets

    # Use modern asyncio client API and avoid deprecated protocol class
    from websockets.asyncio.client import connect as ws_connect
    from typing import Any as WebSocketClientProtocol  # type: ignore
    from websockets.exceptions import ConnectionClosed, WebSocketException
except (
    Exception
):  # pragma: no cover - allow import errors in environments without websockets
    websockets = None  # type: ignore

    def ws_connect(*args, **kwargs):  # type: ignore
        raise ImportError("websockets package is required for CoinApiWebSocketClient")

    WebSocketClientProtocol = object  # type: ignore
    ConnectionClosed = Exception  # type: ignore
    WebSocketException = Exception  # type: ignore


logger = logging.getLogger(__name__)

COINAPI_WS_URL = "wss://ws.coinapi.io/v1/"


def _iso8601_to_unix(ts: str) -> Optional[float]:
    """Convert CoinAPI ISO8601 timestamp to UNIX seconds (float).

    CoinAPI typically returns e.g. "2023-07-27T12:34:56.7890000Z" or without fractional seconds.
    We handle trailing Z and variable fractional precision.
    """
    if not isinstance(ts, str) or not ts:
        return None
    try:
        # Normalize 'Z' to +00:00 for fromisoformat compatibility
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        # Some payloads may include more than 6 microsecond digits; truncate to 6
        if "." in ts:
            date_part, frac_and_zone = ts.split(".", 1)
            # frac_and_zone contains fractional + offset, e.g. 7890000+00:00
            # Split at offset sign (+ or -)
            sign_pos = max(frac_and_zone.find("+"), frac_and_zone.find("-"))
            if sign_pos != -1:
                frac = frac_and_zone[:sign_pos]
                zone = frac_and_zone[sign_pos:]
            else:
                frac = frac_and_zone
                zone = ""
            frac = "".join(c for c in frac if c.isdigit())
            frac = (frac + "000000")[:6]  # pad/truncate to 6
            ts = f"{date_part}.{frac}{zone}"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:  # pragma: no cover - be tolerant to format changes
        try:
            # Fallback: parse without fractions
            base = ts
            if base.endswith("Z"):
                base = base[:-1] + "+00:00"
            dt = datetime.fromisoformat(base)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            logger.debug("Failed to parse ISO8601 timestamp: %s", ts, exc_info=True)
            return None


class CoinApiWebSocketClient:
    """Asynchronous CoinAPI trade WebSocket client with heartbeat and retries.

    Pushes parsed trade ticks into an asyncio.Queue as clean dictionaries:
        {
            "symbol": "BTC",
            "timestamp": 1690461296789,  # milliseconds since epoch (int)
            "price": 29123.45,
            "quantity": 0.0023,
            "raw": {...original payload...}
        }
    """

    def __init__(
        self, symbols: Iterable[str], queue: asyncio.Queue, settings: Settings
    ) -> None:
        if (
            websockets is None
        ):  # pragma: no cover - safety for environments without websockets
            raise ImportError(
                "websockets package is required for CoinApiWebSocketClient"
            )

        self.symbol_tokens: List[str] = [str(s).upper() for s in symbols]
        if not self.symbol_tokens:
            raise ValueError("At least one symbol must be provided")

        if not getattr(settings, "coinapi_api_key", None):
            raise ValueError("CoinAPI API key is missing in settings.coinapi_api_key")

        self.queue: asyncio.Queue = queue
        self.settings: Settings = settings
        self._ws: Optional[WebSocketClientProtocol] = None
        self._stop_event = asyncio.Event()

    @staticmethod
    def _to_symbol_id(token: str) -> str:
        """Construct CoinAPI symbol_id from base token name using config.MASTER_TOKEN_LIST when available.

        Falls back to Binance spot vs USDT format: BINANCE_SPOT_{TOKEN}_USDT
        """
        try:
            from config import MASTER_TOKEN_LIST

            t = str(token).upper()
            symbol = (MASTER_TOKEN_LIST.get(t, {}) or {}).get("coinapi_symbol_id")
            if symbol:
                return symbol
        except Exception:
            pass
        return f"BINANCE_SPOT_{str(token).upper()}_USDT"

    @staticmethod
    def _parse_base_symbol(symbol_id: str) -> Optional[str]:
        """Extract base token from CoinAPI symbol_id, forgivingly.

        Expected formats include BINANCE_SPOT_BTC_USDT or similar underscore-separated.
        We return the token between the venue/market segment and quote (usually index 2).
        """
        if not symbol_id or not isinstance(symbol_id, str):
            return None
        parts = symbol_id.split("_")
        # Try common positions: [EXCHANGE, MARKET, BASE, QUOTE]
        if len(parts) >= 4:
            return parts[-2] if parts[-1] and parts[-2] else None
        if len(parts) >= 3:
            return parts[2]
        return None

    async def _send_hello(self, ws: WebSocketClientProtocol) -> None:
        symbol_ids = [self._to_symbol_id(tok) for tok in self.symbol_tokens]
        hello = {
            "type": "hello",
            "apikey": self.settings.coinapi_api_key,
            "heartbeat": True,
            "subscribe_data_type": ["trade"],
            "subscribe_filter_symbol_id": symbol_ids,
        }
        msg = json.dumps(hello)
        await ws.send(msg)
        logger.info(
            "Sent CoinAPI hello/subscribe message for %d symbols", len(symbol_ids)
        )

    async def run(self) -> None:
        """Run the client indefinitely with exponential backoff and reconnection."""
        backoff = 1.0
        max_backoff = 60.0
        while not self._stop_event.is_set():
            try:
                logger.info("Connecting to CoinAPI websocket: %s", COINAPI_WS_URL)
                self._ws = await ws_connect(
                    COINAPI_WS_URL, ping_interval=20, ping_timeout=20
                )
                await self._send_hello(self._ws)

                backoff = 1.0

                async def _listen(websocket: WebSocketClientProtocol) -> None:
                    async for message in websocket:
                        if self._stop_event.is_set():
                            break
                        try:
                            obj = json.loads(message)
                        except json.JSONDecodeError:
                            logger.debug("Skipping non-JSON message from CoinAPI")
                            continue

                        if not isinstance(obj, dict):
                            continue

                        mtype = obj.get("type")
                        if mtype == "trade":
                            symbol_id = obj.get("symbol_id")
                            t = obj.get("time_exchange")
                            price = obj.get("price")
                            size = obj.get("size")

                            ts_sec = _iso8601_to_unix(t) if isinstance(t, str) else None
                            ts_ms = int(ts_sec * 1000) if ts_sec is not None else None
                            if (
                                symbol_id is None
                                or ts_ms is None
                                or price is None
                                or size is None
                            ):
                                continue
                            try:
                                price_f = float(price)
                                qty_f = float(size)
                            except (TypeError, ValueError):
                                continue

                            base = self._parse_base_symbol(str(symbol_id))
                            if not base:
                                continue

                            trade = {
                                "symbol": base,
                                "timestamp": ts_ms,
                                "price": price_f,
                                "quantity": qty_f,
                                "raw": obj,
                            }
                            await self.queue.put(trade)
                        elif mtype == "heartbeat":
                            logger.debug("CoinAPI heartbeat received")
                        elif mtype == "error":
                            err_msg = obj.get("message") or obj
                            logger.error("CoinAPI error: %s", err_msg)
                            try:
                                await websocket.close()
                            finally:
                                break
                        else:
                            # Other message types can be safely ignored or logged at trace
                            logger.debug("Unhandled CoinAPI message type: %s", mtype)

                await _listen(self._ws)

            except (ConnectionClosed, WebSocketException, OSError) as exc:
                logger.warning(
                    "CoinAPI websocket error: %s. Reconnecting in %.1fs...",
                    exc,
                    backoff,
                )
                await self._close_ws()
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(max_backoff, backoff * 2)
            except asyncio.CancelledError:
                logger.info("CoinAPI websocket task cancelled. Shutting down...")
                await self._close_ws()
                raise
            except Exception as exc:  # pragma: no cover - unexpected
                logger.exception("Unexpected CoinAPI websocket error: %s", exc)
                await self._close_ws()
                await asyncio.sleep(min(5.0, backoff))
                backoff = min(max_backoff, backoff * 2)
            finally:
                if self._stop_event.is_set():
                    await self._close_ws()

    async def stop(self) -> None:
        self._stop_event.set()
        await self._close_ws()

    async def _close_ws(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # pragma: no cover
                logger.debug("Error while closing CoinAPI websocket", exc_info=True)
