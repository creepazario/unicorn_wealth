from sqlalchemy import PrimaryKeyConstraint

from database.models.operational import (
    CurrentPositions,
    TradeLogs,
    TradeSignals,
)


def test_trade_signals_model():
    # Table name
    assert getattr(TradeSignals, "__tablename__", None) == "trade_signals"

    # Required columns/attributes
    required_attrs = [
        "signal_id",
        "timestamp",
        "account_name",
        "token",
        "strategic_directive",
        "prediction_label",
        "avg_probability",
    ]
    for attr in required_attrs:
        assert hasattr(TradeSignals, attr), f"TradeSignals missing attribute: {attr}"


def test_current_positions_model():
    # Table name
    assert getattr(CurrentPositions, "__tablename__", None) == "current_positions"

    # Required columns/attributes
    required_attrs = [
        "account_name",
        "token",
        "trade_id",
        "direction",
        "position_size",
        "entry_price",
        "virtual_stop_loss",
        "virtual_take_profit",
        "entry_timestamp",
    ]
    for attr in required_attrs:
        assert hasattr(
            CurrentPositions, attr
        ), f"CurrentPositions missing attribute: {attr}"

    # Composite primary key verification
    pk = CurrentPositions.__table__.primary_key
    # Ensure it's a PrimaryKeyConstraint instance
    assert isinstance(pk, PrimaryKeyConstraint)

    # Verify it has exactly the expected PK columns
    pk_col_names = {col.name for col in pk.columns}
    assert pk_col_names == {"account_name", "token"}


def test_trade_logs_model():
    # Table name
    assert getattr(TradeLogs, "__tablename__", None) == "trade_logs"

    # Required columns/attributes
    required_attrs = [
        "trade_id",
        "signal_id",
        "account_name",
        "exchange",
        "token",
        "direction",
        "status",
        "entry_exchange_order_id",
        "exit_exchange_order_id",
        "entry_price",
        "exit_price",
        "position_size",
        "pnl_usd",
        "fees",
        "entry_timestamp",
        "exit_timestamp",
    ]
    for attr in required_attrs:
        assert hasattr(TradeLogs, attr), f"TradeLogs missing attribute: {attr}"
