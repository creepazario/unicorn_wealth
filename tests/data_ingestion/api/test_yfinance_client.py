import asyncio

import pandas as pd
import pytest

from unicorn_wealth.data_ingestion.api.yfinance_client import YFinanceClient


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def _no_sleep(duration: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


@pytest.mark.asyncio
async def test_request_formulation_and_parsing(mocker):
    session = mocker.MagicMock()
    client = YFinanceClient(api_key=None, session=session)

    # Build a fake yfinance download DataFrame with DatetimeIndex
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    df_download = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.1, 2.1],
            "Low": [0.9, 1.9],
            "Close": [1.05, 2.05],
            "Volume": [100, 200],
        },
        index=idx,
    )

    mocked = mocker.patch.object(
        client, "_send_request", return_value=df_download, autospec=True
    )

    # No explicit interval -> should default to 1d per client
    df = await client.fetch_data(symbol="BTC-USD", start="2024-01-01", end="2024-01-02")

    assert mocked.call_count == 1
    _, call_args, _ = mocked.mock_calls[0]
    url, params = call_args
    assert url == "yfinance"
    assert params["symbol"] == "BTC-USD"
    assert params["interval"] == "1d"
    assert params["start"] == "2024-01-01"
    assert params["end"] == "2024-01-02"

    # Parsing: DatetimeIndex becomes timestamp column
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]) is True
