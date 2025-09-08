from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import Column, DateTime, Float, MetaData, String, Table
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import Insert
from unittest.mock import AsyncMock

from unicorn_wealth.database.sql_engine import RawDataSQLEngine, FeatureStoreSQLEngine


@pytest.mark.asyncio
async def test_save_data_constructs_correct_upsert_statement(mocker):
    # --- Arrange: Mock AsyncEngine and connection ---
    # Create a mocked connection with an async execute method
    mock_conn = AsyncMock()

    # Create an async context manager for engine.begin()
    class _BeginCtx:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    # Mock AsyncEngine with begin() and sync_engine.url
    mock_engine = mocker.MagicMock()
    mock_engine.begin.return_value = _BeginCtx()
    # Ensure backend detection reports PostgreSQL
    mock_engine.sync_engine = SimpleNamespace(
        url=SimpleNamespace(get_backend_name=lambda: "postgresql")
    )

    # --- Arrange: Create a synthetic table with composite primary key ---
    table_name = "raw_ohlcv"
    meta = MetaData()
    table = Table(
        table_name,
        meta,
        Column("symbol", String, primary_key=True),
        Column("ts", DateTime(timezone=True), primary_key=True),
        Column("open", Float),
        Column("close", Float),
        Column("volume", Float),
    )

    # Patch the internal table loader to return our synthetic table (avoid DB IO)
    mocker.patch.object(
        RawDataSQLEngine, "_ensure_table_loaded", autospec=True, return_value=table
    )

    # --- Arrange: DataFrame matching the table columns ---
    df = pd.DataFrame(
        [
            {
                "symbol": "BTC",
                "ts": pd.Timestamp("2024-01-01T00:00:00Z"),
                "open": 100.0,
                "close": 110.0,
                "volume": 1.5,
            },
            {
                "symbol": "ETH",
                "ts": pd.Timestamp("2024-01-01T00:00:00Z"),
                "open": 10.0,
                "close": 11.0,
                "volume": 15.0,
            },
        ]
    )

    engine = RawDataSQLEngine(mock_engine)

    # --- Act ---
    rowcount = await engine.save_data(df, table_name)

    # --- Assert ---
    # Should attempt to insert/update two rows
    assert rowcount == 2
    # conn.execute should have been called once with an Insert statement
    assert mock_conn.execute.call_count == 1
    stmt = mock_conn.execute.call_args[0][0]

    # Verify type is a PostgreSQL Insert
    assert isinstance(stmt, Insert)

    # Compile SQL for inspection
    compiled_sql = str(
        stmt.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    ).lower()

    # ON CONFLICT clause should exist with the composite PK columns
    assert "on conflict" in compiled_sql
    assert "on conflict (symbol, ts)" in compiled_sql

    # Validate that non-PK columns are present in the SET clause
    pk_cols = {"symbol", "ts"}
    df_cols = set(df.columns)
    update_cols = sorted(c for c in df_cols if c not in pk_cols)
    for col in update_cols:
        expected_piece = f"{col} = excluded.{col}"
        assert expected_piece in compiled_sql


@pytest.mark.asyncio
async def test_feature_store_engine_save_data(mocker):
    # --- Arrange: Mock AsyncEngine and connection ---
    mock_conn = AsyncMock()

    class _BeginCtx:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    mock_engine = mocker.MagicMock()
    mock_engine.begin.return_value = _BeginCtx()
    mock_engine.sync_engine = SimpleNamespace(
        url=SimpleNamespace(get_backend_name=lambda: "postgresql")
    )

    # --- Arrange: Create a synthetic Feature Store table ---
    table_name = "feature_store_4h"
    meta = MetaData()
    table = Table(
        table_name,
        meta,
        Column("timestamp", DateTime(timezone=True), primary_key=True),
        Column("token", String, primary_key=True),
        Column("rsi", Float),
        Column("adx", Float),
    )

    # Ensure the engine uses our synthetic table without touching a real DB
    mocker.patch.object(
        FeatureStoreSQLEngine, "_ensure_table_loaded", autospec=True, return_value=table
    )

    # --- Arrange: sample DataFrame ---
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
                "token": "BTC",
                "rsi": 55.5,
                "adx": 18.2,
            },
            {
                "timestamp": pd.Timestamp("2024-01-01T04:00:00Z"),
                "token": "ETH",
                "rsi": 48.3,
                "adx": 22.7,
            },
        ]
    )

    engine = FeatureStoreSQLEngine(mock_engine)

    # --- Patch filesystem interactions ---
    mocker.patch("pathlib.Path.mkdir")
    to_csv_mock = mocker.patch.object(pd.DataFrame, "to_csv")

    # --- Act ---
    rows = await engine.save_data(df=df, horizon="4h")

    # --- Assert: SQL call ---
    assert rows == 2
    assert mock_conn.execute.call_count == 1
    stmt = mock_conn.execute.call_args[0][0]
    assert isinstance(stmt, Insert)

    compiled_sql = str(
        stmt.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    ).lower()

    assert "insert into" in compiled_sql
    assert table_name in compiled_sql
    assert "on conflict (timestamp, token)" in compiled_sql

    # Non-PK fields should be updated on conflict
    for col in ["rsi", "adx"]:
        assert f"{col} = excluded.{col}" in compiled_sql

    # --- Assert: CSV export call ---
    assert to_csv_mock.call_count == 1
    args, kwargs = to_csv_mock.call_args
    assert args[0] == "output/training_data/feature_store_4h.csv"
    assert kwargs.get("index") is False
