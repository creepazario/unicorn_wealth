import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from unicorn_wealth.database.models.base import get_async_engine
from unicorn_wealth.core.config_loader import Settings

_ = pytest  # mark as used to satisfy flake8


def _dummy_settings(**overrides):
    # Provide all required Settings fields with dummy values, allowing overrides.
    base = {
        # logging
        "log_level": "INFO",
        "log_file_path": "logs/test.log",
        # database
        "database_url": "sqlite+aiosqlite:///:memory:",
        "db_pool_size": 5,
        "db_max_overflow": 10,
        "db_pool_recycle_seconds": 3600,
        # data provider api keys
        "coinapi_api_key": "x",
        "santiment_api_key": "x",
        "coinmarketcap_api_key": "x",
        "finnhub_api_key": "x",
        # telegram
        "telegram_api_id": "x",
        "telegram_api_hash": "x",
        "telegram_bot_token": "x",
        "telegram_admin_channel_id": "x",
        "telegram_trade_channel_id": "x",
        # binance
        "binance_unicorn_api_key": "x",
        "binance_unicorn_api_secret": "x",
        "binance_personal_api_key": "x",
        "binance_personal_api_secret": "x",
        # kraken
        "kraken_unicorn_api_key": "x",
        "kraken_unicorn_api_secret": "x",
        "kraken_personal_api_key": "x",
        "kraken_personal_api_secret": "x",
        # hyperliquid
        "hyperliquid_unicorn_api_key": "x",
        "hyperliquid_unicorn_api_secret": "x",
        "hyperliquid_personal_api_key": "x",
        "hyperliquid_personal_api_secret": "x",
        # bitget
        "bitget_unicorn_api_key": "x",
        "bitget_unicorn_api_secret": "x",
        "bitget_unicorn_api_passphrase": "x",
        "bitget_personal_api_key": "x",
        "bitget_personal_api_secret": "x",
        "bitget_personal_api_passphrase": "x",
        # bybit
        "bybit_unicorn_api_key": "x",
        "bybit_unicorn_api_secret": "x",
        "bybit_personal_api_key": "x",
        "bybit_personal_api_secret": "x",
        # kucoin
        "kucoin_unicorn_api_key": "x",
        "kucoin_unicorn_api_secret": "x",
        "kucoin_unicorn_api_passphrase": "x",
        "kucoin_personal_api_key": "x",
        "kucoin_personal_api_secret": "x",
        "kucoin_personal_api_passphrase": "x",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_get_async_engine_applies_pool_settings(monkeypatch):
    # Capture kwargs passed to create_async_engine inside our module
    captured: dict = {}

    from sqlalchemy.ext.asyncio import create_async_engine as real_create_async_engine

    def fake_create_async_engine(url, **kwargs):  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured["kwargs"] = dict(kwargs)
        # Avoid sqlite aiosqlite pool kwargs errors by stripping them for the real call
        safe_kwargs = dict(kwargs)
        for k in (
            "pool_size",
            "max_overflow",
            "pool_recycle",
            "poolclass",
            "connect_args",
        ):
            safe_kwargs.pop(k, None)
        return real_create_async_engine(url, **safe_kwargs)

    # Patch the symbol used by get_async_engine
    monkeypatch.setattr(
        "unicorn_wealth.database.models.base.create_async_engine",
        fake_create_async_engine,
    )

    settings = _dummy_settings(
        database_url="sqlite+aiosqlite:///:memory:",
        db_pool_size=5,
        db_max_overflow=10,
        db_pool_recycle_seconds=3600,
    )

    engine = get_async_engine(settings)
    try:
        # Verify type
        assert isinstance(engine, AsyncEngine)
        # Verify our factory forwarded pool settings correctly
        assert captured["kwargs"]["pool_size"] == 5
        assert captured["kwargs"]["max_overflow"] == 10
        assert captured["kwargs"]["pool_recycle"] == 3600
    finally:
        engine.sync_engine.dispose()
