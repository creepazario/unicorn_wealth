from __future__ import annotations

from typing import Optional

# Resolve logging defaults regardless of import style (package or flat)
try:
    from config import (
        LOG_LEVEL as DEFAULT_LOG_LEVEL,
        LOG_FILE_PATH as DEFAULT_LOG_FILE_PATH,
    )  # type: ignore
except Exception:  # pragma: no cover - fallback when importing with packages
    try:
        from . import config as _cfg  # type: ignore

        DEFAULT_LOG_LEVEL = _cfg.LOG_LEVEL  # type: ignore[attr-defined]
        DEFAULT_LOG_FILE_PATH = _cfg.LOG_FILE_PATH  # type: ignore[attr-defined]
    except Exception:
        from unicorn_wealth.config import (
            LOG_LEVEL as DEFAULT_LOG_LEVEL,
            LOG_FILE_PATH as DEFAULT_LOG_FILE_PATH,
        )  # type: ignore

# Pydantic v2 deprecated BaseSettings in core;
# prefer v1-compat path if needed.
try:  # pragma: no cover - primary path (pydantic v1)
    from pydantic import (
        BaseModel,
        BaseSettings,
        root_validator,
    )  # type: ignore[attr-defined]

    _SETTINGS_BACKEND = "pydantic_v1"
except Exception:  # pragma: no cover - pydantic v2
    try:
        # Use pydantic v1 compatibility layer for both BaseModel and BaseSettings
        from pydantic.v1 import (
            BaseModel,
            BaseSettings,
            root_validator,
        )  # type: ignore

        _SETTINGS_BACKEND = "pydantic_v1_compat"
    except Exception:
        # Last resort: pydantic-settings (works with pydantic v2 BaseModel)
        from pydantic import BaseModel  # type: ignore
        from pydantic_settings import BaseSettings  # type: ignore

        # Define a shim for root_validator in v2 context.
        # Using model_post_init pattern is overkill here.
        # Tests rely on v1 behavior via compat. If reached,
        # nested fields won't be auto-built.
        def root_validator(*args, **kwargs):  # type: ignore
            def decorator(fn):
                return fn

            return decorator

        _SETTINGS_BACKEND = "pydantic_settings"


class AccountCredentials(BaseModel):
    """Credentials for a single account on an exchange."""

    api_key: str
    api_secret: str
    api_passphrase: Optional[str] = None


class ExchangeSettings(BaseModel):
    """Holds credentials for both account types for a single exchange."""

    unicorn: AccountCredentials
    personal: AccountCredentials


class Settings(BaseSettings):
    """Application settings loaded from environment variables (.env).

    Notes:
    - Flat fields for each .env variable
      (clear 1:1 mapping, type-safe access).
    - Provides nested ExchangeSettings as computed attributes
      to enforce centralized configurability and avoid
      ad-hoc dict access patterns.
    """

    # --- Logging ---
    log_level: str = DEFAULT_LOG_LEVEL
    log_file_path: str = DEFAULT_LOG_FILE_PATH

    # --- Data Provider API Keys ---
    coinapi_api_key: str
    santiment_api_key: str
    coinmarketcap_api_key: str
    finnhub_api_key: str

    # --- Database ---
    database_url: str

    # --- Telegram ---
    telegram_api_id: str
    telegram_api_hash: str
    telegram_bot_token: str
    telegram_admin_channel_id: str
    telegram_trade_channel_id: str

    # --- Exchange API Keys (Flat mapping to .env) ---
    # Binance
    binance_unicorn_api_key: str
    binance_unicorn_api_secret: str
    binance_personal_api_key: str
    binance_personal_api_secret: str

    # Kraken
    kraken_unicorn_api_key: str
    kraken_unicorn_api_secret: str
    kraken_personal_api_key: str
    kraken_personal_api_secret: str

    # HyperLiquid
    hyperliquid_unicorn_api_key: str
    hyperliquid_unicorn_api_secret: str
    hyperliquid_personal_api_key: str
    hyperliquid_personal_api_secret: str

    # Bitget (includes passphrase)
    bitget_unicorn_api_key: str
    bitget_unicorn_api_secret: str
    bitget_unicorn_api_passphrase: str
    bitget_personal_api_key: str
    bitget_personal_api_secret: str
    bitget_personal_api_passphrase: str

    # Bybit
    bybit_unicorn_api_key: str
    bybit_unicorn_api_secret: str
    bybit_personal_api_key: str
    bybit_personal_api_secret: str

    # Kucoin (includes passphrase)
    kucoin_unicorn_api_key: str
    kucoin_unicorn_api_secret: str
    kucoin_unicorn_api_passphrase: str
    kucoin_personal_api_key: str
    kucoin_personal_api_secret: str
    kucoin_personal_api_passphrase: str

    # ------------------------------------------------------------------
    # Computed, nested models for exchanges
    # ------------------------------------------------------------------
    @property
    def binance(self) -> ExchangeSettings:
        return ExchangeSettings(
            unicorn=AccountCredentials(
                api_key=self.binance_unicorn_api_key,
                api_secret=self.binance_unicorn_api_secret,
            ),
            personal=AccountCredentials(
                api_key=self.binance_personal_api_key,
                api_secret=self.binance_personal_api_secret,
            ),
        )

    @property
    def kraken(self) -> ExchangeSettings:
        return ExchangeSettings(
            unicorn=AccountCredentials(
                api_key=self.kraken_unicorn_api_key,
                api_secret=self.kraken_unicorn_api_secret,
            ),
            personal=AccountCredentials(
                api_key=self.kraken_personal_api_key,
                api_secret=self.kraken_personal_api_secret,
            ),
        )

    @property
    def hyperliquid(self) -> ExchangeSettings:
        return ExchangeSettings(
            unicorn=AccountCredentials(
                api_key=self.hyperliquid_unicorn_api_key,
                api_secret=self.hyperliquid_unicorn_api_secret,
            ),
            personal=AccountCredentials(
                api_key=self.hyperliquid_personal_api_key,
                api_secret=self.hyperliquid_personal_api_secret,
            ),
        )

    @property
    def bitget(self) -> ExchangeSettings:
        return ExchangeSettings(
            unicorn=AccountCredentials(
                api_key=self.bitget_unicorn_api_key,
                api_secret=self.bitget_unicorn_api_secret,
                api_passphrase=self.bitget_unicorn_api_passphrase,
            ),
            personal=AccountCredentials(
                api_key=self.bitget_personal_api_key,
                api_secret=self.bitget_personal_api_secret,
                api_passphrase=self.bitget_personal_api_passphrase,
            ),
        )

    @property
    def bybit(self) -> ExchangeSettings:
        return ExchangeSettings(
            unicorn=AccountCredentials(
                api_key=self.bybit_unicorn_api_key,
                api_secret=self.bybit_unicorn_api_secret,
            ),
            personal=AccountCredentials(
                api_key=self.bybit_personal_api_key,
                api_secret=self.bybit_personal_api_secret,
            ),
        )

    @property
    def kucoin(self) -> ExchangeSettings:
        return ExchangeSettings(
            unicorn=AccountCredentials(
                api_key=self.kucoin_unicorn_api_key,
                api_secret=self.kucoin_unicorn_api_secret,
                api_passphrase=self.kucoin_unicorn_api_passphrase,
            ),
            personal=AccountCredentials(
                api_key=self.kucoin_personal_api_key,
                api_secret=self.kucoin_personal_api_secret,
                api_passphrase=self.kucoin_personal_api_passphrase,
            ),
        )

    class Config:
        env_file = ".env"


def load_settings() -> Settings:
    """Load and validate settings from environment variables (.env)."""
    return Settings()  # type: ignore[call-arg]
