from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil

try:  # optional at runtime; handle absence gracefully
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - environment may not have NVML
    pynvml = None  # type: ignore

from config import (
    DATA_FRESHNESS_THRESHOLD_SECONDS,
    MAX_CPU_PERCENT,
    MAX_GPU_TEMPERATURE,
    MAX_RAM_PERCENT,
)
from core.dataframe_registry import DataFrameRegistry


class SystemMonitor:
    """Continuously checks technical health of the app and server.

    Orchestrates three checks:
      - System resources (CPU, RAM, GPU temperature)
      - External API heartbeats via clients' ping()
      - Data freshness for key DataFrames via DataFrameRegistry

    If any check fails or breaches thresholds, a consolidated alert is
    sent to the ADMIN channel using the provided TelegramNotifier.
    """

    def __init__(
        self,
        settings: Any,
        notifier: Any,
        api_clients: Dict[str, Any],
        dataframe_registry: DataFrameRegistry,
    ) -> None:
        self.settings = settings
        self.notifier = notifier
        self.api_clients = api_clients or {}
        self.df_registry = dataframe_registry

    async def run_all_checks(self) -> None:
        """Run all checks and send a single consolidated alert if needed.

        This method is resilient: it runs all checks even if one fails,
        and collates all error messages into a single system alert.
        """
        checks = [
            self._check_system_resources(),
            self._check_api_heartbeats(),
            self._check_data_freshness(),
        ]

        results: List[Optional[str]] = await asyncio.gather(*checks, return_exceptions=True)  # type: ignore

        errors: List[str] = []
        for res in results:
            if isinstance(res, Exception):
                errors.append(f"Unexpected check error: {res}")
            elif isinstance(res, str) and res.strip():
                errors.append(res)

        if errors:
            # Consolidated alert
            message = "\n".join(f"- {e}" for e in errors)
            try:
                self.notifier.send_notification(
                    type="SYSTEM_ALERT",
                    data={
                        "timestamp": datetime.now(timezone.utc),
                        "title": "System Health Checks Failed",
                        "message": message,
                        "severity": "CRITICAL",
                    },
                    channel_type="ADMIN",
                )
            except Exception:
                # Last-resort: do not raise in scheduler loop
                pass

    # ---- Private checks ----
    async def _check_system_resources(self) -> str | None:
        """Check CPU, RAM, and GPU temperature against thresholds.

        Returns an error string upon breach, else None.
        """
        try:
            cpu_pct = float(psutil.cpu_percent(interval=0.1))
            ram_pct = float(psutil.virtual_memory().percent)
        except Exception as e:
            return f"System resource read failed: {e}"

        errors: List[str] = []
        if cpu_pct > float(MAX_CPU_PERCENT):
            errors.append(
                f"CPU usage high: {cpu_pct:.1f}% > {float(MAX_CPU_PERCENT):.1f}%"
            )
        if ram_pct > float(MAX_RAM_PERCENT):
            errors.append(
                f"RAM usage high: {ram_pct:.1f}% > {float(MAX_RAM_PERCENT):.1f}%"
            )

        # GPU temperature via NVML if available
        try:
            if pynvml is not None:
                try:
                    pynvml.nvmlInit()  # type: ignore[attr-defined]
                    device_count = pynvml.nvmlDeviceGetCount()  # type: ignore[attr-defined]
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # type: ignore[attr-defined]
                        temps = pynvml.nvmlDeviceGetTemperature(
                            handle, getattr(pynvml, "NVML_TEMPERATURE_GPU", 0)
                        )  # type: ignore[attr-defined]
                        if float(temps) > float(MAX_GPU_TEMPERATURE):
                            name = "GPU"
                            try:
                                name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            errors.append(
                                f"{name} temperature high: {float(temps):.1f}°C > {float(MAX_GPU_TEMPERATURE):.1f}°C"
                            )
                finally:
                    try:
                        pynvml.nvmlShutdown()  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception as e:
            # NVML failure should not break monitor; report as warning
            errors.append(f"GPU temperature check failed: {e}")

        if errors:
            return "System resources alert:\n" + "\n".join(errors)
        return None

    async def _check_api_heartbeats(self) -> str | None:
        """Ping all configured API clients. Returns error message if any fail."""
        if not self.api_clients:
            return None

        async def _ping_one(name: str, client: Any) -> tuple[str, Optional[str]]:
            try:
                ping = getattr(client, "ping", None)
                if ping is None:
                    return name, f"Client '{name}' has no ping() method"
                result = ping()
                if asyncio.iscoroutine(result):
                    result = await result
                # Interpret truthy as success; can be response dict or bool
                if not result:
                    return name, f"Ping failed for '{name}'"
                return name, None
            except Exception as e:
                return name, f"Ping exception for '{name}': {e}"

        tasks = [_ping_one(n, c) for n, c in self.api_clients.items()]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        failures = [msg for _n, msg in results if msg]
        if failures:
            return "API heartbeat issues:\n" + "\n".join(f"• {m}" for m in failures)
        return None

    async def _check_data_freshness(self) -> str | None:
        """Verify key DataFrame recency against threshold.

        Uses a primary key 'BTC_ohlcv_1m_df' if present; otherwise, attempts to
        find a DataFrame with 'ohlcv' or '1m' in its name as fallback.
        """
        name_candidates = ["BTC_ohlcv_1m_df"]
        try:
            names = await self.df_registry.list_dfs()
            # Add heuristic candidates only if primary isn't present
            if "BTC_ohlcv_1m_df" not in names:
                for nm in names:
                    s = nm.lower()
                    if ("ohlcv" in s or "1m" in s) and nm not in name_candidates:
                        name_candidates.append(nm)
        except Exception as e:
            return f"DataFrameRegistry access failed: {e}"

        last_ts: Optional[datetime] = None
        chosen_name: Optional[str] = None
        for nm in name_candidates:
            try:
                df = await self.df_registry.get_df(nm)
                if df is None or df.empty:
                    continue
                # Try to locate timestamp column
                ts_col = None
                for candidate in ("timestamp", "time", "datetime", "date"):
                    if candidate in df.columns:
                        ts_col = candidate
                        break
                if ts_col is None:
                    continue
                ts_series = df[ts_col]
                try:
                    ts = ts_series.iloc[-1]
                    # Normalize to datetime
                    if not isinstance(ts, datetime):
                        ts = datetime.fromisoformat(str(ts))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                last_ts = ts
                chosen_name = nm
                break
            except KeyError:
                continue
            except Exception as e:  # unexpected df issues
                return f"Data freshness retrieval failed for '{nm}': {e}"

        if last_ts is None:
            return "No suitable DataFrame found for freshness check"

        now = datetime.now(timezone.utc)
        age_seconds = (now - last_ts).total_seconds()
        if age_seconds > float(DATA_FRESHNESS_THRESHOLD_SECONDS):
            return (
                "Data freshness issue: "
                f"'{chosen_name}' last update {int(age_seconds)}s ago > "
                f"{int(float(DATA_FRESHNESS_THRESHOLD_SECONDS))}s"
            )
        return None
