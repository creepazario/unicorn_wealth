import sys
from pathlib import Path

# Ensure project root is on sys.path for `import core`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_loader import load_settings  # noqa: E402


def test_load_settings_basic_fields():
    s = load_settings()
    assert s.coinapi_api_key == "your_amberdata_key"
    assert s.santiment_api_key == "your_santiment_key"
    assert s.coinmarketcap_api_key == "your_cmc_key"
    assert s.finnhub_api_key == "your_finnhub_key"

    assert s.database_url.startswith("postgresql+asyncpg://")

    assert s.telegram_api_id == "your_api_id"
    assert s.telegram_api_hash == "your_api_hash"
    assert s.telegram_bot_token == "your_bot_token"
    assert s.telegram_admin_channel_id == "your_admin_channel_id"
    assert s.telegram_trade_channel_id == "your_trade_channel_id"


def test_nested_exchange_credentials():
    s = load_settings()

    # Binance (no passphrase)
    assert s.binance.unicorn.api_key == "..."
    assert s.binance.unicorn.api_secret == "..."
    assert s.binance.unicorn.api_passphrase is None
    assert s.binance.personal.api_key == "..."
    assert s.binance.personal.api_secret == "..."
    assert s.binance.personal.api_passphrase is None

    # Bitget (with passphrase)
    assert s.bitget.unicorn.api_key == "..."
    assert s.bitget.unicorn.api_secret == "..."
    assert s.bitget.unicorn.api_passphrase == "..."
    assert s.bitget.personal.api_key == "..."
    assert s.bitget.personal.api_secret == "..."
    assert s.bitget.personal.api_passphrase == "..."

    # Kucoin (with passphrase)
    assert s.kucoin.unicorn.api_key == "..."
    assert s.kucoin.unicorn.api_secret == "..."
    assert s.kucoin.unicorn.api_passphrase == "..."
    assert s.kucoin.personal.api_key == "..."
    assert s.kucoin.personal.api_secret == "..."
    assert s.kucoin.personal.api_passphrase == "..."
