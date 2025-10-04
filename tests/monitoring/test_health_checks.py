from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict

import pandas as pd
import pytest

from monitoring.health_checks import SystemMonitor


@pytest.fixture
def settings() -> Any:
    # minimal settings namespace; SystemMonitor uses none directly
    return SimpleNamespace()


class DummyNotifier:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def send_notification(
        self, *, type: str, data: Dict[str, Any], channel_type: str
    ) -> None:  # noqa: A002 - arg name from lib
        self.calls.append({"type": type, "data": data, "channel_type": channel_type})


class DummyDFRegistry:
    def __init__(self) -> None:
        self._dfs: Dict[str, pd.DataFrame] = {}

    async def get_df(self, name: str) -> pd.DataFrame:
        if name not in self._dfs:
            raise KeyError(name)
        return self._dfs[name]

    async def list_dfs(self) -> list[str]:
        return list(self._dfs.keys())

    def set_df(self, name: str, df: pd.DataFrame) -> None:
        self._dfs[name] = df


@pytest.mark.asyncio
async def test_resources_ok(monkeypatch):
    # psutil below thresholds
    class VM:
        percent = 50.0

    monkeypatch.setattr("psutil.cpu_percent", lambda interval=0.1: 10.0)
    monkeypatch.setattr("psutil.virtual_memory", lambda: VM)

    # Mock pynvml to report cool GPU
    class NVML:
        _inited = False

        @staticmethod
        def nvmlInit():
            NVML._inited = True

        @staticmethod
        def nvmlDeviceGetCount():
            return 1

        @staticmethod
        def nvmlDeviceGetHandleByIndex(i):  # noqa: ARG001 - signature compat
            return object()

        @staticmethod
        def nvmlDeviceGetTemperature(handle, which):  # noqa: ARG001
            return 40  # below threshold

        @staticmethod
        def nvmlDeviceGetName(handle):  # noqa: ARG001
            return b"MockGPU"

        @staticmethod
        def nvmlShutdown():
            NVML._inited = False

        NVML_TEMPERATURE_GPU = 0

    import sys

    monkeypatch.setitem(sys.modules, "pynvml", NVML)

    notifier = DummyNotifier()
    dfreg = DummyDFRegistry()
    # Fresh data within threshold
    now = datetime.now(timezone.utc)
    dfreg.set_df("BTC_ohlcv_1m_df", pd.DataFrame({"timestamp": [now]}))

    monitor = SystemMonitor(settings=SimpleNamespace(), notifier=notifier, api_clients={}, dataframe_registry=dfreg)  # type: ignore[arg-type]

    await monitor.run_all_checks()

    # No alert should be sent
    assert notifier.calls == []


@pytest.mark.asyncio
async def test_ram_exceeded(monkeypatch):
    # RAM above threshold
    class VM:
        percent = 95.0

    monkeypatch.setattr("psutil.cpu_percent", lambda interval=0.1: 10.0)
    monkeypatch.setattr("psutil.virtual_memory", lambda: VM)

    # Ensure pynvml import path exists but unused
    class NVML:
        @staticmethod
        def nvmlInit():
            pass

        @staticmethod
        def nvmlDeviceGetCount():
            return 0

        @staticmethod
        def nvmlShutdown():
            pass

    monkeypatch.setitem(__import__("sys").modules, "pynvml", NVML)

    notifier = DummyNotifier()
    dfreg = DummyDFRegistry()
    now = datetime.now(timezone.utc)
    dfreg.set_df("BTC_ohlcv_1m_df", pd.DataFrame({"timestamp": [now]}))

    monitor = SystemMonitor(settings=SimpleNamespace(), notifier=notifier, api_clients={}, dataframe_registry=dfreg)  # type: ignore[arg-type]

    await monitor.run_all_checks()

    # An alert should be sent mentioning RAM
    assert len(notifier.calls) == 1
    msg = notifier.calls[0]["data"]["message"]
    assert "RAM" in msg or "Ram" in msg or "ram" in msg


@pytest.mark.asyncio
async def test_api_heartbeat_fails(monkeypatch):
    # Healthy resources
    class VM:
        percent = 10.0

    monkeypatch.setattr("psutil.cpu_percent", lambda interval=0.1: 10.0)
    monkeypatch.setattr("psutil.virtual_memory", lambda: VM)

    class NVML:
        @staticmethod
        def nvmlInit():
            pass

        @staticmethod
        def nvmlDeviceGetCount():
            return 0

        @staticmethod
        def nvmlShutdown():
            pass

    monkeypatch.setitem(__import__("sys").modules, "pynvml", NVML)

    notifier = DummyNotifier()
    dfreg = DummyDFRegistry()
    now = datetime.now(timezone.utc)
    dfreg.set_df("BTC_ohlcv_1m_df", pd.DataFrame({"timestamp": [now]}))

    class BadClient:
        def ping(self):
            raise RuntimeError("boom")

    monitor = SystemMonitor(
        settings=SimpleNamespace(),
        notifier=notifier,
        api_clients={"bad": BadClient()},
        dataframe_registry=dfreg,
    )  # type: ignore[arg-type]

    await monitor.run_all_checks()

    assert len(notifier.calls) == 1
    msg = notifier.calls[0]["data"]["message"]
    assert "API" in msg or "Ping" in msg or "heartbeat" in msg.lower()


@pytest.mark.asyncio
async def test_data_is_stale(monkeypatch):
    # Healthy resources
    class VM:
        percent = 10.0

    monkeypatch.setattr("psutil.cpu_percent", lambda interval=0.1: 10.0)
    monkeypatch.setattr("psutil.virtual_memory", lambda: VM)

    class NVML:
        @staticmethod
        def nvmlInit():
            pass

        @staticmethod
        def nvmlDeviceGetCount():
            return 0

        @staticmethod
        def nvmlShutdown():
            pass

    monkeypatch.setitem(__import__("sys").modules, "pynvml", NVML)

    notifier = DummyNotifier()
    dfreg = DummyDFRegistry()
    # Create stale timestamp older than threshold (1800s)
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=3600)
    dfreg.set_df("BTC_ohlcv_1m_df", pd.DataFrame({"timestamp": [stale_ts]}))

    monitor = SystemMonitor(settings=SimpleNamespace(), notifier=notifier, api_clients={}, dataframe_registry=dfreg)  # type: ignore[arg-type]

    await monitor.run_all_checks()

    assert len(notifier.calls) == 1
    msg = notifier.calls[0]["data"]["message"]
    assert (
        "fresh" in msg.lower() or "stale" in msg.lower() or "last update" in msg.lower()
    )
