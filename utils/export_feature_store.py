from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from core.config_loader import load_settings
from database.models.base import get_async_engine


TABLE_NAME = "feature_store_1h"


async def _get_non_all_null_columns(
    engine: AsyncEngine, schema: Optional[str] = None
) -> List[str]:
    """Return column names from TABLE_NAME excluding columns where all values are NULL.

    Strategy: fetch all column names from information_schema, then execute a single
    SELECT COUNT(col) aggregate per column to identify columns having at least
    one non-null value. COUNT(col) counts only non-null values in SQL.
    """
    tbl = TABLE_NAME
    sch = schema or "public"

    # 1) Fetch column names in ordinal order
    async with engine.begin() as conn:
        cols_result = await conn.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ordinal_position
                """
            ),
            {"schema": sch, "table": tbl},
        )
        all_columns = [row[0] for row in cols_result]

    if not all_columns:
        raise ValueError(f"Table '{sch}.{tbl}' has no columns or does not exist.")

    # 2) Build a single row query with COUNT(col) for each column.
    #    If a column is of a type that cannot be counted directly (uncommon), COUNT(col)
    #    still works in Postgres for all types since it just counts non-NULL occurrences.
    count_selects = ", ".join([f'COUNT("{c}") AS "{c}"' for c in all_columns])
    count_sql = f'SELECT {count_selects} FROM "{sch}"."{tbl}"'

    async with engine.begin() as conn:
        counts_result = await conn.execute(text(count_sql))
        counts_row = counts_result.first()
        if counts_row is None:
            # Empty table - treat all columns as all-null (so none selected)
            return []

    non_all_null_cols: List[str] = []
    for c in all_columns:
        cnt = counts_row._mapping[c]
        if cnt and int(cnt) > 0:
            non_all_null_cols.append(c)

    return non_all_null_cols


async def _export_table_to_csv(
    engine: AsyncEngine,
    output_path: Path | str,
    columns: Sequence[str],
    schema: Optional[str] = None,
    where: Optional[str] = None,
    chunk_size: int = 100_000,
) -> Path:
    """Stream the table rows to CSV selecting only provided columns.

    Parameters
    ----------
    engine: AsyncEngine
        Async SQLAlchemy engine bound to the target DB.
    output_path: Path | str
        File path where the CSV will be written.
    columns: Sequence[str]
        Column names to include in export.
    schema: Optional[str]
        Schema name (defaults to 'public').
    where: Optional[str]
        Optional raw SQL WHERE clause (without the 'WHERE' keyword).
    chunk_size: int
        Number of rows to fetch per chunk.
    """
    if not columns:
        # Nothing to export
        p = Path(output_path)
        # Create empty CSV with header? Requirement didn't specify; create an empty file.
        p.write_text("")
        return p

    sch = schema or "public"
    cols_sql = ", ".join([f'"{c}"' for c in columns])

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_written = False
    offset = 0

    async with engine.begin() as conn:
        while True:
            where_clause = f" WHERE {where}" if where else ""
            limit_sql = (
                f'SELECT {cols_sql} FROM "{sch}"."{TABLE_NAME}"{where_clause}'
                f" OFFSET {offset} LIMIT {chunk_size}"
            )

            result = await conn.execute(text(limit_sql))
            rows = result.fetchall()
            if not rows:
                break

            df = pd.DataFrame.from_records(rows, columns=list(columns))
            # Write CSV incrementally
            df.to_csv(out_path, mode="a", header=not header_written, index=False)
            header_written = True

            fetched = len(rows)
            offset += fetched
            if fetched < chunk_size:
                break

    return out_path


def export_feature_store_1h_to_csv(
    output_path: Path | str,
    *,
    schema: Optional[str] = None,
    where: Optional[str] = None,
    chunk_size: int = 100_000,
) -> Path:
    """Export the contents of the Postgres table 'feature_store_1h' to CSV.

    - Excludes any columns where all values are NULL.
    - Streams data in chunks to avoid high memory usage.

    Parameters
    ----------
    output_path: Path | str
        Destination CSV filepath.
    schema: Optional[str]
        Optional schema name (defaults to 'public').
    where: Optional[str]
        Optional raw SQL WHERE clause (without 'WHERE'), e.g., "symbol = 'BTC'".
    chunk_size: int
        Number of rows per fetch batch (default 100k).
    """
    settings = load_settings()
    engine = get_async_engine(settings)

    async def _run() -> Path:
        cols = await _get_non_all_null_columns(engine, schema)
        return await _export_table_to_csv(
            engine, output_path, cols, schema=schema, where=where, chunk_size=chunk_size
        )

    return asyncio.run(_run())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export feature_store_1h table to CSV, excluding all-null columns."
    )
    parser.add_argument(
        "--output",
        default="output/feature_store_1h.csv",
        help="Output CSV file path (default: output/feature_store_1h.csv)",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Optional schema name (defaults to 'public')",
    )
    parser.add_argument(
        "--where",
        default=None,
        help="Optional raw SQL WHERE clause without the 'WHERE' keyword",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows to fetch per chunk",
    )

    args = parser.parse_args()
    path = export_feature_store_1h_to_csv(
        args.output, schema=args.schema, where=args.where, chunk_size=args.chunk_size
    )
    print(f"Exported to: {path}")
