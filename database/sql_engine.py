from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

try:  # Prefer relative import when used inside the package
    from .models.base import Base
except Exception:  # pragma: no cover - fallback when imported from project root
    from unicorn_wealth.database.models.base import Base  # type: ignore


class RawDataSQLEngine:
    """Asynchronous SQL engine for persisting raw pandas DataFrames with upsert.

    This class is designed for dependency injection: callers provide an existing
    SQLAlchemy AsyncEngine. It dynamically resolves the target table by name
    using SQLAlchemy metadata and performs an upsert (insert on conflict update)
    using PostgreSQL's ON CONFLICT clause.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        """Initialize the engine wrapper.

        Parameters
        ----------
        engine: AsyncEngine
            An initialized SQLAlchemy asynchronous engine.
        """
        self._engine = engine

    async def _ensure_table_loaded(self, table_name: str) -> Table:
        """Ensure the table is available in Base.metadata, reflecting if needed.

        Reflection is executed in a synchronous function via `run_sync` as
        required by SQLAlchemy's async API.
        """
        if table_name not in Base.metadata.tables:
            # Reflect only the requested table into Base.metadata
            async with self._engine.begin() as conn:

                def _reflect(sync_conn: Any) -> None:
                    Base.metadata.reflect(bind=sync_conn, only=[table_name])

                await conn.run_sync(_reflect)

        table = Base.metadata.tables.get(table_name)
        if table is None:
            # As a fallback, try reflecting into a fresh MetaData
            # (e.g., if Base is unused)
            meta = MetaData()
            async with self._engine.begin() as conn:

                def _reflect_fresh(sync_conn: Any) -> None:
                    meta.reflect(bind=sync_conn, only=[table_name])

                await conn.run_sync(_reflect_fresh)
            table = meta.tables.get(table_name)

        if table is None:
            raise ValueError(f"Table '{table_name}' not found in metadata or database.")
        return table

    async def _upsert_dataframe(self, df: pd.DataFrame, table_name: str) -> int:
        """Common upsert implementation reused by engines.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the data to persist.
        table_name: str
            Name of the target table.
        """
        if df is None or df.empty:
            return 0

        # Ensure table is available
        table = await self._ensure_table_loaded(table_name)

        # Verify we are using a PostgreSQL engine (required for pg_insert)
        backend_name = None
        try:
            # Prefer sync_engine.url for broad SQLAlchemy compatibility
            sync_eng = getattr(self._engine, "sync_engine", None) or self._engine
            url = getattr(sync_eng, "url", None)
            if url is not None:
                backend_name = url.get_backend_name()
        except Exception:  # pragma: no cover - be permissive in detection
            backend_name = None
        if backend_name != "postgresql":
            raise NotImplementedError(
                "Upsert currently supports only PostgreSQL backends."
            )

        # Filter DataFrame to columns present in the table
        table_cols = {col.name for col in table.columns}
        df_cols = [c for c in df.columns if c in table_cols]
        if not df_cols:
            raise ValueError(
                "None of the DataFrame columns match the target table columns."
            )

        # Convert DataFrame to list of dictionaries suitable for insertion
        records: List[Dict[str, Any]] = (
            df[df_cols].where(pd.notnull(df[df_cols]), None).to_dict(orient="records")
        )
        if not records:
            return 0

        # Determine primary key columns for conflict target
        pk_cols = [col.name for col in table.primary_key.columns]
        if not pk_cols:
            raise ValueError(
                f"Table '{table_name}' does not have a primary key; upsert requires PK."
            )

        # Build insert statement with ON CONFLICT DO UPDATE
        insert_stmt = pg_insert(table).values(records)

        # Non-PK columns to update on conflict (only those present in the DF)
        update_columns = [c for c in df_cols if c not in pk_cols]
        if not update_columns:
            # If only PKs are provided, ON CONFLICT DO NOTHING is the safest choice
            upsert_stmt = insert_stmt.on_conflict_do_nothing(index_elements=pk_cols)
        else:
            set_mapping = {
                col: getattr(insert_stmt.excluded, col) for col in update_columns
            }
            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=pk_cols,
                set_=set_mapping,
            )

        # Execute asynchronously
        async with self._engine.begin() as conn:
            await conn.execute(upsert_stmt)

        return len(records)

    async def save_data(self, df: pd.DataFrame, table_name: str) -> int:
        """Persist a DataFrame into the database with upsert behavior.

        The upsert uses PostgreSQL's ON CONFLICT with the table's primary key
        columns as the conflict target. Non-PK columns are updated from the
        incoming row on conflict.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the data to persist.
        table_name: str
            Name of the target table.

        Returns
        -------
        int
            Number of rows attempted to insert/update (0 for empty DataFrame).
        """
        return await self._upsert_dataframe(df, table_name)


class FeatureStoreSQLEngine(RawDataSQLEngine):
    """Engine to persist wide feature DataFrames and export to CSV.

    This engine writes to horizon-specific Feature Store tables with an upsert
    on the composite primary key (timestamp, token), and then exports the same
    DataFrame to a CSV file for offline inspection.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)

    async def save_data(self, df: pd.DataFrame, horizon: str) -> int:
        """Save the feature DataFrame to SQL and export to CSV.

        Parameters
        ----------
        df: pd.DataFrame
            The wide feature DataFrame to save.
        horizon: str
            Horizon label like '1h', '4h', or '8h'.

        Returns
        -------
        int
            Number of rows attempted to insert/update (0 for empty DataFrame).
        """
        if not isinstance(horizon, str) or not horizon:
            raise ValueError("horizon must be a non-empty string, e.g., '1h'.")

        table_name = f"feature_store_{horizon}"

        # 1) Upsert into the feature store table
        affected = await self._upsert_dataframe(df, table_name)

        # 2) Export to CSV after successful DB operation
        output_dir = Path("output") / "training_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"feature_store_{horizon}.csv"
        # Ensure pandas does not write the index to the CSV
        if df is not None and not df.empty:
            df.to_csv(csv_path.as_posix(), index=False)

        return affected
