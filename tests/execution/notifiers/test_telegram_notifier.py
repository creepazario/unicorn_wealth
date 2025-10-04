from types import SimpleNamespace

import pytest

from execution.notifiers.telegram_notifier import TelegramNotifier


@pytest.mark.asyncio
async def test_sends_position_opened_notification(mocker):
    # Patch Telethon client class
    client_cls = mocker.patch(
        "execution.notifiers.telegram_notifier.TelegramClient", autospec=True
    )
    client_instance = client_cls.return_value
    client_instance.start = mocker.AsyncMock()
    client_instance.send_message = mocker.AsyncMock()

    # Prepare settings with all enabled
    settings = SimpleNamespace(
        telegram_api_id="api_id",
        telegram_api_hash="api_hash",
        telegram_bot_token="bot_token",
        telegram_admin_channel_id="-100123",
        telegram_trade_channel_id="-100456",
        TELEGRAM_ENABLED=True,
        NOTIFICATION_SETTINGS={
            "ON_POSITION_OPEN": True,
            "ON_POSITION_CLOSE": True,
            "SYSTEM_ALERTS": True,
        },
    )

    notifier = TelegramNotifier(settings)

    sample = {
        "timestamp": 1_700_000_000,
        "exchange": "Binance",
        "account": "unicorn",
        "token": "BTC",
        "direction": "LONG",
        "entry_price": 50000.0,
        "position_size": 0.25,
    }

    await notifier.send_notification(
        type="ON_POSITION_OPEN", data=sample, channel_type="TRADE"
    )

    # Assert start called then send_message called once with correct channel and message content
    client_instance.start.assert_awaited_once_with(bot_token="bot_token")
    client_instance.send_message.assert_awaited_once()
    args, kwargs = client_instance.send_message.await_args
    # Check channel
    assert kwargs.get("entity") == "-100456"
    # Check message contains key fields
    msg = kwargs.get("message")
    assert "BTC" in msg
    assert "Entry" in msg and "$50,000.00" in msg
    assert "Size" in msg and "0.2500" in msg
    assert kwargs.get("parse_mode") == "md"


@pytest.mark.asyncio
async def test_notification_disabled_by_master_switch(mocker):
    client_cls = mocker.patch(
        "execution.notifiers.telegram_notifier.TelegramClient", autospec=True
    )
    client_instance = client_cls.return_value

    settings = SimpleNamespace(
        telegram_api_id="api_id",
        telegram_api_hash="api_hash",
        telegram_bot_token="bot_token",
        telegram_admin_channel_id="-100123",
        telegram_trade_channel_id="-100456",
        TELEGRAM_ENABLED=False,
        NOTIFICATION_SETTINGS={"ON_POSITION_OPEN": True},
    )

    notifier = TelegramNotifier(settings)

    await notifier.send_notification(
        type="ON_POSITION_OPEN", data={"token": "BTC"}, channel_type="TRADE"
    )

    client_instance.start.assert_not_called()
    client_instance.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notification_disabled_by_type_switch(mocker):
    client_cls = mocker.patch(
        "execution.notifiers.telegram_notifier.TelegramClient", autospec=True
    )
    client_instance = client_cls.return_value

    settings = SimpleNamespace(
        telegram_api_id="api_id",
        telegram_api_hash="api_hash",
        telegram_bot_token="bot_token",
        telegram_admin_channel_id="-100123",
        telegram_trade_channel_id="-100456",
        TELEGRAM_ENABLED=True,
        NOTIFICATION_SETTINGS={"ON_POSITION_OPEN": False},
    )

    notifier = TelegramNotifier(settings)

    await notifier.send_notification(
        type="ON_POSITION_OPEN", data={"token": "BTC"}, channel_type="TRADE"
    )

    client_instance.start.assert_not_called()
    client_instance.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_sends_to_correct_channel(mocker):
    client_cls = mocker.patch(
        "execution.notifiers.telegram_notifier.TelegramClient", autospec=True
    )
    client_instance = client_cls.return_value
    client_instance.start = mocker.AsyncMock()
    client_instance.send_message = mocker.AsyncMock()

    settings = SimpleNamespace(
        telegram_api_id="api_id",
        telegram_api_hash="api_hash",
        telegram_bot_token="bot_token",
        telegram_admin_channel_id="-100999",
        telegram_trade_channel_id="-100888",
        TELEGRAM_ENABLED=True,
        NOTIFICATION_SETTINGS={"SYSTEM_ALERTS": True, "ON_POSITION_OPEN": True},
    )

    notifier = TelegramNotifier(settings)

    # Send to TRADE
    await notifier.send_notification(
        type="ON_POSITION_OPEN",
        data={"token": "ETH", "entry_price": 2000.0, "position_size": 1.0},
        channel_type="TRADE",
    )
    # Send to ADMIN
    await notifier.send_notification(
        type="SYSTEM_ALERTS",
        data={"title": "Health", "message": "OK"},
        channel_type="ADMIN",
    )

    # Two sends total
    assert client_instance.send_message.await_count == 2

    # Collect channel ids from calls
    calls = [
        kwargs.get("entity")
        for args, kwargs in client_instance.send_message.await_args_list
    ]
    assert calls[0] == "-100888"
    assert calls[1] == "-100999"
