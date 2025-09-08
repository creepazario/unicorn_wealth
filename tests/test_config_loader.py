import sys
from pathlib import Path

# Ensure project root is on sys.path for `import core`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_loader import load_settings  # noqa: E402


def test_load_settings_basic_fields():
    s = load_settings()
    # API keys should be present as non-empty strings
    # (avoid asserting specific secret values)
    assert isinstance(s.coinapi_api_key, str) and s.coinapi_api_key
    assert isinstance(s.santiment_api_key, str) and s.santiment_api_key
    assert isinstance(s.coinmarketcap_api_key, str) and s.coinmarketcap_api_key
    assert isinstance(s.finnhub_api_key, str) and s.finnhub_api_key

    # DB URL format
    assert isinstance(s.database_url, str)
    assert s.database_url.startswith("postgresql+asyncpg://")

    # Telegram settings should be present as non-empty strings
    assert isinstance(s.telegram_api_id, str) and s.telegram_api_id
    assert isinstance(s.telegram_api_hash, str) and s.telegram_api_hash
    assert isinstance(s.telegram_bot_token, str) and s.telegram_bot_token
    assert isinstance(s.telegram_admin_channel_id, str) and s.telegram_admin_channel_id
    assert isinstance(s.telegram_trade_channel_id, str) and s.telegram_trade_channel_id


def test_nested_exchange_credentials():
    s = load_settings()

    # Binance (no passphrase)
    assert isinstance(s.binance.unicorn.api_key, str) and s.binance.unicorn.api_key
    assert (
        isinstance(s.binance.unicorn.api_secret, str) and s.binance.unicorn.api_secret
    )
    assert s.binance.unicorn.api_passphrase is None
    assert isinstance(s.binance.personal.api_key, str) and s.binance.personal.api_key
    assert (
        isinstance(s.binance.personal.api_secret, str) and s.binance.personal.api_secret
    )
    assert s.binance.personal.api_passphrase is None

    # Bitget (with passphrase) - unicorn may be unset in some environments
    unicorn_key = s.bitget.unicorn.api_key
    unicorn_secret = s.bitget.unicorn.api_secret
    unicorn_pass = s.bitget.unicorn.api_passphrase
    assert (unicorn_key is None) or (isinstance(unicorn_key, str) and unicorn_key)
    assert (unicorn_secret is None) or (
        isinstance(unicorn_secret, str) and unicorn_secret
    )
    assert (unicorn_pass is None) or (isinstance(unicorn_pass, str) and unicorn_pass)

    # Personal should be present
    assert isinstance(s.bitget.personal.api_key, str) and s.bitget.personal.api_key
    assert (
        isinstance(s.bitget.personal.api_secret, str) and s.bitget.personal.api_secret
    )
    assert (
        isinstance(s.bitget.personal.api_passphrase, str)
        and s.bitget.personal.api_passphrase
    )

    # Kucoin (with passphrase) - unicorn may be unset
    ku_unicorn_key = s.kucoin.unicorn.api_key
    ku_unicorn_secret = s.kucoin.unicorn.api_secret
    ku_unicorn_pass = s.kucoin.unicorn.api_passphrase
    assert (ku_unicorn_key is None) or (
        isinstance(ku_unicorn_key, str) and ku_unicorn_key
    )
    assert (ku_unicorn_secret is None) or (
        isinstance(ku_unicorn_secret, str) and ku_unicorn_secret
    )
    assert (ku_unicorn_pass is None) or (
        isinstance(ku_unicorn_pass, str) and ku_unicorn_pass
    )

    # Personal should be present
    assert isinstance(s.kucoin.personal.api_key, str) and s.kucoin.personal.api_key
    assert (
        isinstance(s.kucoin.personal.api_secret, str) and s.kucoin.personal.api_secret
    )
    assert (
        isinstance(s.kucoin.personal.api_passphrase, str)
        and s.kucoin.personal.api_passphrase
    )
