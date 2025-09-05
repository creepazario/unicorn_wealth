from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# Ensure project root is on sys.path for module resolution
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import all models so Alembic can discover them for autogenerate
try:
    from database.models.base import Base
    from database.models import (  # noqa: F401
        raw_data as _raw_data,
        feature_stores as _feature_stores,
        operational as _operational,
    )
except Exception:  # pragma: no cover - fallback when running as package
    from unicorn_wealth.database.models.base import Base  # type: ignore
    from unicorn_wealth.database.models import (  # type: ignore # noqa: F401
        raw_data as _raw_data,
        feature_stores as _feature_stores,
        operational as _operational,
    )

# this is the Alembic Config object, which provides access to the values
# within the .ini file in use.
config = context.config

# Allow overriding database URL from environment (.env) when alembic.ini
# contains a placeholder. This lets operators keep secrets in .env.
try:  # pragma: no cover - optional
    import os
    from dotenv import load_dotenv

    load_dotenv()
    env_db_url = os.getenv("DATABASE_URL")
    ini_db_url = config.get_main_option("sqlalchemy.url")
    if env_db_url and (
        ini_db_url.strip() == "postgresql+asyncpg://user:password@host:port/dbname"
        or "user:password@host:port" in ini_db_url
    ):
        config.set_main_option("sqlalchemy.url", env_db_url)
except Exception:
    pass

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine, though an
    Engine is acceptable here as well. By skipping the Engine creation we don't
    even need a DBAPI to be available.
    """

    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode using async engine."""

    url = config.get_main_option("sqlalchemy.url")
    connectable = create_async_engine(url)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    import asyncio

    asyncio.run(run_migrations_online())
