from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Try to load TimescaleDB SQLAlchemy plugin to register hypertable args if available
try:  # pragma: no cover - only needed when package is installed
    import sqlalchemy_timescaledb  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - safely ignore if not installed
    pass

try:  # Prefer async-compatible pool when available
    from sqlalchemy.ext.asyncio.pool import (
        AsyncAdaptedQueuePool as _AsyncQueuePool,
    )  # type: ignore
except Exception:  # pragma: no cover - fallback name in some versions
    _AsyncQueuePool = None  # type: ignore

from core.config_loader import Settings


class Base(DeclarativeBase):
    pass


def get_async_engine(settings: Settings) -> AsyncEngine:
    """Create and return an asynchronous SQLAlchemy engine.

    The engine uses the database URL and pooling parameters provided by the
    Settings instance. No global engine is created; callers are responsible
    for managing the engine lifecycle.
    """
    kwargs = {
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_recycle": settings.db_pool_recycle_seconds,
    }
    # Use asyncio-compatible queue pool when available so pool settings are honored
    if _AsyncQueuePool is not None:  # type: ignore[truthy-bool]
        kwargs["poolclass"] = _AsyncQueuePool  # type: ignore[assignment]

    db_url = settings.database_url
    # For in-memory sqlite, use a shared-cache URI so pooling works in tests
    if db_url.startswith("sqlite+aiosqlite://") and ":memory:" in db_url:
        db_url = "sqlite+aiosqlite:///file:memdb1?mode=memory&cache=shared&uri=true"
        kwargs["connect_args"] = {"uri": True}

    return create_async_engine(
        db_url,
        **kwargs,
    )
