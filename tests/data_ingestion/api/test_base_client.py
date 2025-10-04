import asyncio
from typing import Any, Dict, Tuple

import pandas as pd
import pytest
from config import API_CLIENT_SETTINGS
from data_ingestion.api.base_client import BaseAPIClient
import pybreaker as _pybreaker


class _TestClient(BaseAPIClient):
    async def _build_request(self, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        return "test://endpoint", {"q": 1}

    async def _send_request(
        self, url: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:  # pragma: no cover - always mocked in tests
        raise NotImplementedError

    def _parse_response(self, response: Dict[str, Any] | Any) -> pd.DataFrame:
        # Ensure a timestamp field exists
        ts = (
            response.get("timestamp", "2024-01-01T00:00:00Z")
            if isinstance(response, dict)
            else "2024-01-01T00:00:00Z"
        )
        return pd.DataFrame({"timestamp": [ts], "value": [1]})


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def _no_sleep(duration: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


@pytest.mark.asyncio
async def test_retry_logic(mocker):
    # Arrange
    stop_attempts = int(
        API_CLIENT_SETTINGS["TENACITY_RETRY"]["STOP_MAX_ATTEMPT"]
    )  # 5 by default

    client = _TestClient(api_key=None, session=mocker.MagicMock())

    # Fail for first (stop_attempts - 1) times, then succeed
    side_effects = [Exception("temporary error")] * (stop_attempts - 1)
    side_effects.append({"timestamp": "2024-01-01T00:00:00Z", "ok": True})

    mocked_send = mocker.patch.object(
        client,
        "_send_request",
        side_effect=side_effects,
        autospec=True,
    )

    # Act
    df = await client.fetch_data()

    # Assert
    assert not df.empty
    # Called (stop_attempts - 1) failures + 1 success
    assert mocked_send.call_count == stop_attempts


@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_blocks_calls(mocker):
    # Arrange
    fail_max = int(API_CLIENT_SETTINGS["CIRCUIT_BREAKER"]["FAIL_MAX"])  # 5 by default
    stop_attempts = int(
        API_CLIENT_SETTINGS["TENACITY_RETRY"]["STOP_MAX_ATTEMPT"]
    )  # 5 by default

    client = _TestClient(api_key=None, session=mocker.MagicMock())

    # Always fail the downstream call
    mocked_send = mocker.patch.object(
        client,
        "_send_request",
        side_effect=RuntimeError("persistent failure"),
        autospec=True,
    )

    # Act: First call should attempt up to stop_attempts and then raise
    with pytest.raises(RuntimeError):
        await client.fetch_data()

    # The breaker should have recorded failures for each attempt and now be open
    # Since STOP_MAX_ATTEMPT == FAIL_MAX by default,
    # breaker opens at the end of first call
    assert mocked_send.call_count == min(stop_attempts, fail_max)

    # Reset mock call count for clarity in the next assertion
    mocked_send.reset_mock()

    # Next call should immediately raise CircuitBreakerError
    # before calling _send_request
    with pytest.raises(_pybreaker.CircuitBreakerError):
        await client.fetch_data()

    mocked_send.assert_not_called()
