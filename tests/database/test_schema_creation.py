from typing import Set

import pytest
import sqlalchemy as sa

try:
    from core.config_loader import load_settings
    from database.models.base import Base, get_async_engine
    from database.models import (  # noqa: F401
        raw_data as _raw_data,
        feature_stores as _feature_stores,
        operational as _operational,
    )
except Exception:  # pragma: no cover - fallback when running as package
    from unicorn_wealth.core.config_loader import load_settings  # type: ignore
    from unicorn_wealth.database.models.base import (  # type: ignore
        Base,
        get_async_engine,
    )
    from unicorn_wealth.database.models import (  # type: ignore # noqa: F401
        raw_data as _raw_data,
        feature_stores as _feature_stores,
        operational as _operational,
    )


@pytest.mark.asyncio
async def test_all_tables_are_created() -> None:
    """Verify that all SQLAlchemy models' tables exist in the live database.

    This test connects to the actual database specified by DATABASE_URL and
    inspects the public schema to list existing tables, then compares them
    against the tables defined by our ORM models (Base.metadata).
    """
    settings = load_settings()
    engine = get_async_engine(settings)

    try:
        async with engine.connect() as conn:

            def _get_table_names(sync_conn) -> Set[str]:
                inspector = sa.inspect(sync_conn)
                names = set(inspector.get_table_names())
                # Ignore Alembic's versioning table when comparing to ORM models
                names.discard("alembic_version")
                return names

            existing_tables = await conn.run_sync(_get_table_names)
    finally:
        await engine.dispose()

    expected_tables = set(Base.metadata.tables.keys())

    assert existing_tables == expected_tables
