import builtins
import os
from unittest.mock import mock_open

from core.config_loader import load_settings


def test_load_settings_successfully(mocker):
    # Fake .env content with at least the required variables from the issue, plus
    # dummy values for other required Settings fields to satisfy validation.
    env_content = "\n".join(
        [
            'DATABASE_URL="test_db_url"',
            'COINAPI_API_KEY="test_coinapi_key"',
            'SANTIMENT_API_KEY="x"',
            'COINMARKETCAP_API_KEY="x"',
            'FINNHUB_API_KEY="x"',
            'TELEGRAM_API_ID="x"',
            'TELEGRAM_API_HASH="x"',
            'TELEGRAM_BOT_TOKEN="x"',
            'TELEGRAM_ADMIN_CHANNEL_ID="-1001234567890"',
            'TELEGRAM_TRADE_CHANNEL_ID="-1009876543210"',
            # Binance (asserted below)
            'BINANCE_UNICORN_API_KEY="test_binance_key"',
            'BINANCE_UNICORN_API_SECRET="test_binance_secret"',
            'BINANCE_PERSONAL_API_KEY="test_binance_personal_key"',
            'BINANCE_PERSONAL_API_SECRET="test_binance_personal_secret"',
            # Kraken
            'KRAKEN_UNICORN_API_KEY="x"',
            'KRAKEN_UNICORN_API_SECRET="x"',
            'KRAKEN_PERSONAL_API_KEY="x"',
            'KRAKEN_PERSONAL_API_SECRET="x"',
            # HyperLiquid
            'HYPERLIQUID_UNICORN_API_KEY="x"',
            'HYPERLIQUID_UNICORN_API_SECRET="x"',
            'HYPERLIQUID_PERSONAL_API_KEY="x"',
            'HYPERLIQUID_PERSONAL_API_SECRET="x"',
            # Bitget (with passphrase)
            'BITGET_UNICORN_API_KEY="x"',
            'BITGET_UNICORN_API_SECRET="x"',
            'BITGET_UNICORN_API_PASSPHRASE="x"',
            'BITGET_PERSONAL_API_KEY="x"',
            'BITGET_PERSONAL_API_SECRET="x"',
            'BITGET_PERSONAL_API_PASSPHRASE="x"',
            # Bybit
            'BYBIT_UNICORN_API_KEY="x"',
            'BYBIT_UNICORN_API_SECRET="x"',
            'BYBIT_PERSONAL_API_KEY="x"',
            'BYBIT_PERSONAL_API_SECRET="x"',
            # Kucoin (with passphrase)
            'KUCOIN_UNICORN_API_KEY="x"',
            'KUCOIN_UNICORN_API_SECRET="x"',
            'KUCOIN_UNICORN_API_PASSPHRASE="x"',
            'KUCOIN_PERSONAL_API_KEY="x"',
            'KUCOIN_PERSONAL_API_SECRET="x"',
            'KUCOIN_PERSONAL_API_PASSPHRASE="x"',
        ]
    )

    # Mock open to return our fake .env content
    # and os.path.exists to pretend the file exists
    m = mock_open(read_data=env_content)
    mocker.patch.object(builtins, "open", m)
    mocker.patch.object(os.path, "exists", return_value=True)

    # Execute
    settings = load_settings()

    # Assertions (flat fields)
    assert settings.database_url == "test_db_url"
    assert settings.coinapi_api_key == "test_coinapi_key"

    # Assertions (nested structure)
    assert settings.binance.unicorn.api_key == "test_binance_key"
    assert settings.binance.unicorn.api_secret == "test_binance_secret"
    assert settings.binance.personal.api_key == "test_binance_personal_key"
    assert settings.binance.personal.api_secret == "test_binance_personal_secret"
