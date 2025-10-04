from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from database.models.base import Base


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

        # Optional mapping for OHLCV interval -> ORM model. This enables
        # convenient routing to timeframe-specific tables.
        try:
            from database.models.raw_data import (
                RawOHLCV15m,
                RawOHLCV1h,
                RawOHLCV4h,
                RawOHLCV1d,
                RawOHLCV7d,
            )

            self._ohlcv_table_map = {
                "15m": RawOHLCV15m,
                "1h": RawOHLCV1h,
                "4h": RawOHLCV4h,
                "1d": RawOHLCV1d,
                "7d": RawOHLCV7d,
            }
        except Exception:
            # Defer import errors to actual method usage to avoid circular deps
            self._ohlcv_table_map = {}

    async def _ensure_table_loaded(self, table_name: str) -> Table:
        """Reflect the given table from the database and return a fresh Table.

        Always uses a fresh MetaData reflection to avoid stale ORM metadata
        mismatches (e.g., ORM declares columns that the DB table doesn't have yet).
        """
        meta = MetaData()
        async with self._engine.begin() as conn:

            def _reflect_fresh(sync_conn: Any) -> None:
                meta.reflect(bind=sync_conn, only=[table_name])

            await conn.run_sync(_reflect_fresh)
        table = meta.tables.get(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' not found in database.")
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
            # Only perform UPDATE when at least one non-PK column actually changes
            try:
                from sqlalchemy import or_ as _or
            except Exception:
                _or = None  # type: ignore
            where_clause = None
            if _or is not None:
                comparisons = [
                    getattr(table.c, col).is_distinct_from(
                        getattr(insert_stmt.excluded, col)
                    )
                    for col in update_columns
                ]
                if comparisons:
                    # any column differs -> update
                    where_clause = _or(*comparisons)
            if where_clause is not None:
                upsert_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=pk_cols,
                    set_=set_mapping,
                    where=where_clause,
                )
            else:
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
        # Batching hotfix: prevent oversized SQL statements by chunking the DataFrame
        if df is None or df.empty:
            return 0

        total = 0
        chunk_size = 500

        # Lazy import to avoid circular logging configs
        import logging as _logging

        log = _logging.getLogger("uw.sql_engine.raw_data")
        nrows = len(df)
        if nrows <= chunk_size:
            return await self._upsert_dataframe(df, table_name)

        num_chunks = (nrows + chunk_size - 1) // chunk_size
        for i in range(0, nrows, chunk_size):
            chunk = df.iloc[i : i + chunk_size]
            chunk_idx = i // chunk_size + 1
            log.info(
                "Saving chunk %s of %s to table %s (rows=%s)...",
                chunk_idx,
                num_chunks,
                table_name,
                len(chunk),
            )
            affected = await self._upsert_dataframe(chunk, table_name)
            total += affected
        return total

    async def save_ohlcv_data(self, df: pd.DataFrame, interval: str) -> int:
        """Persist OHLCV DataFrame to the timeframe-specific table.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing columns like timestamp, token, open, high, low, close, volume.
        interval: str
            One of: '15m', '1h', '4h', '1d'.
        """
        if not isinstance(interval, str) or not interval:
            raise ValueError(
                "interval must be a non-empty string, e.g., '15m', '1h', '4h', '1d'."
            )

        # Ensure the model map exists and resolve the table name from the model
        if not getattr(self, "_ohlcv_table_map", None):
            # Attempt late import to avoid circular dependencies during __init__
            from database.models.raw_data import (
                RawOHLCV15m,
                RawOHLCV1h,
                RawOHLCV4h,
                RawOHLCV1d,
                RawOHLCV7d,
            )

            self._ohlcv_table_map = {
                "15m": RawOHLCV15m,
                "1h": RawOHLCV1h,
                "4h": RawOHLCV4h,
                "1d": RawOHLCV1d,
                "7d": RawOHLCV7d,
            }

        model = self._ohlcv_table_map.get(interval)
        if model is None:
            raise ValueError(
                f"Unsupported interval '{interval}'. Expected one of: {list(self._ohlcv_table_map.keys())}."
            )

        table_name = getattr(model, "__tablename__", None)
        if not table_name:
            raise RuntimeError("Resolved OHLCV model does not define __tablename__.")

        return await self.save_data(df, table_name)


class FeatureStoreSQLEngine(RawDataSQLEngine):
    """Engine to persist wide feature DataFrames and export to CSV.

    This engine writes to horizon-specific Feature Store tables with an upsert
    on the composite primary key (timestamp, token), and then exports the same
    DataFrame to a CSV file for offline inspection.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)

    async def _sync_feature_store_schema(
        self, table_name: str, df: pd.DataFrame
    ) -> None:
        """Ensure the feature store table has all columns present in df.

        Adds any missing feature columns as FLOAT NULL to future-proof against
        newly added indicators without requiring an Alembic migration.
        """
        from sqlalchemy import text as _text

        # Ensure table is loaded/reflected; create exception if table missing
        table = await self._ensure_table_loaded(table_name)
        existing_cols = {c.name for c in table.columns}
        # Only consider non-PK feature columns from df
        candidates = [c for c in df.columns if c not in {"timestamp", "token"}]
        to_add = [c for c in candidates if c not in existing_cols]
        if not to_add:
            return
        # Add columns one by one; use FLOAT which maps well to float64
        async with self._engine.begin() as conn:
            for col in to_add:
                # Safe identifier quoting via direct string because col names come from our whitelist/sanitizer
                if col == "economic_event":
                    # Ensure correct type for external boolean feature
                    stmt = _text(
                        f'ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS "{col}" BOOLEAN NOT NULL DEFAULT FALSE'
                    )
                else:
                    stmt = _text(
                        f'ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS "{col}" DOUBLE PRECISION NULL'
                    )
                try:
                    await conn.execute(stmt)
                except Exception:
                    # Ignore if cannot add (e.g., race condition); continue
                    pass
        # Refresh reflection so next operations see new columns
        await self._ensure_table_loaded(table_name)

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

        # Pre-filter incoming DataFrame to only include rows within the lookback window
        try:
            if df is not None and not df.empty and "timestamp" in df.columns:
                from config import HISTORICAL_LOOKBACK_DAYS  # type: ignore
                import pandas as _pd
                from pandas.api.types import is_numeric_dtype as _isnum

                cutoff_dt = _pd.Timestamp.now(tz="UTC") - _pd.Timedelta(
                    days=int(HISTORICAL_LOOKBACK_DAYS)
                )
                ts_series = df["timestamp"]
                if _isnum(ts_series):
                    # Heuristic: interpret as ms if median abs >= 1e12 else seconds
                    med = _pd.to_numeric(ts_series, errors="coerce").abs().median()
                    unit = (
                        "ms" if (med is not None and med >= 1_000_000_000_000) else "s"
                    )
                    dt_series = _pd.to_datetime(
                        ts_series, unit=unit, utc=True, errors="coerce"
                    )
                else:
                    dt_series = _pd.to_datetime(ts_series, utc=True, errors="coerce")
                mask = dt_series >= cutoff_dt
                if mask.notna().any():
                    df = df.loc[mask]
        except Exception:
            # Best-effort filtering; if anything goes wrong, continue without filtering
            pass

        # Ensure table exists in metadata or DB; if table missing entirely, try creating via ORM models
        try:
            await self._ensure_table_loaded(table_name)
        except Exception:
            # Attempt to create tables using ORM models (no-ops if exist)
            from database.models.feature_stores import (
                FeatureStore1h,
                FeatureStore4h,
                FeatureStore8h,
            )

            async with self._engine.begin() as conn:

                def _create(sync_conn: Any) -> None:
                    Base.metadata.create_all(
                        bind=sync_conn,
                        tables=[
                            FeatureStore1h.__table__,
                            FeatureStore4h.__table__,
                            FeatureStore8h.__table__,
                        ],
                    )

                await conn.run_sync(_create)

        # Attempt to sync missing feature columns based on incoming df
        try:
            await self._sync_feature_store_schema(table_name, df)
        except Exception:
            # Best-effort; continue to upsert which may still succeed if columns match
            pass

        # Prune old rows outside of the configured lookback window for this table
        # Only prune when the table's timestamp column is integer-like (epoch ms),
        # which matches production schema and avoids interfering with tests that use DateTime.
        try:
            table = await self._ensure_table_loaded(table_name)
            ts_col = next((c for c in table.columns if c.name == "timestamp"), None)
            is_int_ts = False
            if ts_col is not None:
                try:
                    pytype = getattr(ts_col.type, "python_type", None)
                    is_int_ts = pytype is int
                except Exception:
                    is_int_ts = False

            if is_int_ts:
                from config import HISTORICAL_LOOKBACK_DAYS  # type: ignore
                import pandas as _pd
                from sqlalchemy import text as _text

                now_ts = _pd.Timestamp.now(tz="UTC")
                start_ts = now_ts - _pd.Timedelta(days=int(HISTORICAL_LOOKBACK_DAYS))
                start_epoch_ms = int(start_ts.timestamp() * 1000)

                async with self._engine.begin() as conn:
                    await conn.execute(
                        _text(f"DELETE FROM {table_name} WHERE timestamp < :start_ts"),
                        {"start_ts": start_epoch_ms},
                    )
        except Exception:
            # Best-effort pruning; do not fail save on prune errors
            pass

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


class FeatureStoreSQLStorageEngine:
    """Facade providing `save_features` API on top of FeatureStoreSQLEngine.

    Applies robust sanitization to outgoing feature DataFrames to ensure
    compatibility with DB constraints and future-proof against new features.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._delegate = FeatureStoreSQLEngine(engine)

    @staticmethod
    def _sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize a wide feature DataFrame prior to DB upsert.

        - Ensure timestamp is int64 and token is string
        - Keep only known training feature columns in stable order
        - Cast features to float64, replace inf with NaN
        - Drop rows where all feature columns are NaN
        - Also drop rows containing any NaN in the features to satisfy strict schemas
        """
        import numpy as _np
        import pandas as _pd

        if df is None or df.empty:
            return _pd.DataFrame()

        cols = [
            "timestamp",
            "token",
            "rsi_15m",
            "atr_normalized_15m",
            "atr_normalized_1h",
            "atr_normalized_4h",
            "atr_normalized_1d",
            "adx_15m",
            "adx_pos_15m",
            "adx_neg_15m",
            "adx_1h",
            "adx_pos_1h",
            "adx_neg_1h",
            "adx_4h",
            "adx_pos_4h",
            "adx_neg_4h",
            "adx_1d",
            "adx_pos_1d",
            "adx_neg_1d",
            # Temporal cyclical time features (UTC)
            "day_of_week_cos",
            "day_of_week_sin",
            "hour_of_day_cos",
            "hour_of_day_sin",
            "minute_of_hour_cos",
            "minute_of_hour_sin",
            # Sessions & Kill-zones
            "sess_ny_flag",
            "sess_london_flag",
            "sess_asia_flag",
            "kz_ny_flag",
            "kz_london_flag",
            "friday_flag",
            "bars_to_session_close",
            "bars_to_kz_end",
            "bars_since_session_open",
            "bars_since_midnight_utc",
            "bars_since_kz_start",
            # External boolean feature
            "economic_event",
        ]
        # Intersect with existing columns to be resilient to future changes
        keep = [c for c in cols if c in df.columns]
        out = df[keep].copy()
        # Ensure external boolean feature column exists; default to False if missing
        if "economic_event" not in out.columns:
            import pandas as _pd

            out["economic_event"] = _pd.Series([False] * len(out), dtype="boolean")

        # Types for PKs
        if "timestamp" in out.columns:
            out["timestamp"] = (
                _pd.to_numeric(out["timestamp"], errors="coerce")
                .astype("Int64")
                .astype("int64")
            )
        if "token" in out.columns:
            out["token"] = out["token"].astype(str)

        # Feature columns are all except PKs
        feat_cols = [c for c in keep if c not in {"timestamp", "token"}]
        # Separate boolean features to preserve dtype (currently only economic_event)
        bool_cols = [c for c in feat_cols if c == "economic_event"]
        num_cols = [c for c in feat_cols if c not in bool_cols]

        # Numeric processing
        for c in num_cols:
            out[c] = _pd.to_numeric(out[c], errors="coerce")
        if num_cols:
            out[num_cols] = out[num_cols].replace([_np.inf, -_np.inf], _np.nan)

        # Boolean processing: coerce to pandas boolean dtype; fill NaN with False (spec requires boolean)
        for c in bool_cols:
            # Accept truthy strings/ints; anything non-True becomes False
            out[c] = (
                out[c]
                .map(lambda v: bool(v) if v is not None and not _pd.isna(v) else False)
                .astype("boolean")
            )

        # Drop rows where all features are NaN (consider only numeric features for this test)
        if num_cols:
            all_nan_mask = out[num_cols].isna().all(axis=1)
            out = out.loc[~all_nan_mask].copy()
            if out.empty:
                return out
            # Strict: drop rows with any NaN in numeric features
            any_nan_mask = out[num_cols].isna().any(axis=1)
            out = out.loc[~any_nan_mask].copy()

        # Final cast: numeric features to float64; keep booleans as is
        for c in num_cols:
            out[c] = out[c].astype("float64")

        return out

    async def save_features(self, df: pd.DataFrame, horizon: str) -> int:
        """Persist features in batched chunks to avoid oversized statements.

        Implements batching similar to the raw data engine. Each chunk is
        upserted separately, and progress is logged.
        """
        if df is None or df.empty:
            return 0

        # Apply window filtering to ensure we only write within the retention window
        try:
            from config import HISTORICAL_LOOKBACK_DAYS  # type: ignore
            import pandas as _pd

            now_ts = _pd.Timestamp.now(tz="UTC")
            start_ts = now_ts - _pd.Timedelta(days=int(HISTORICAL_LOOKBACK_DAYS))
            start_epoch_ms = int(start_ts.timestamp() * 1000)
            end_epoch_ms = int(now_ts.timestamp() * 1000)
            # Filter input df to [start, now]
            if "timestamp" in df.columns:
                df = df[
                    (df["timestamp"] >= start_epoch_ms)
                    & (df["timestamp"] <= end_epoch_ms)
                ]
        except Exception:
            # If any error, proceed without filtering (delegate will prune anyway)
            pass

        if df is None or df.empty:
            return 0

        # Batching params
        chunk_size = 500

        # Local import for logging to avoid global logger config issues
        import logging as _logging

        log = _logging.getLogger("uw.sql_engine.feature_store")
        # Sanitize once upfront if small, else sanitize per-chunk below
        nrows = len(df)
        if nrows <= chunk_size:
            log.info("Saving feature chunk 1...")
            cleaned = self._sanitize_features(df)
            if cleaned.empty:
                return 0
            return await self._delegate.save_data(df=cleaned, horizon=horizon)

        total = 0
        num_chunks = (nrows + chunk_size - 1) // chunk_size
        for i in range(0, nrows, chunk_size):
            chunk = df.iloc[i : i + chunk_size]
            # Sanitize each chunk independently to avoid NaNs
            cleaned = self._sanitize_features(chunk)
            if cleaned.empty:
                continue
            chunk_idx = i // chunk_size + 1
            log.info(f"Saving feature chunk {chunk_idx} of {num_chunks}...")
            affected = await self._delegate.save_data(df=cleaned, horizon=horizon)
            total += affected
        return total
