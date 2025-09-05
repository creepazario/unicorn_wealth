"""Operational ORM models for live trading system.

This module defines TimescaleDB-backed tables and related operational tables
used by the live system to track model signals, current open positions, and an
immutable audit trail of completed trades.
"""

from __future__ import annotations

import uuid

from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import BigInteger, Float, Integer, String

from .base import Base


# TimescaleDB hypertable configuration helper, mirroring other modules
TIMESCALEDB_ARGS = ({"timescaledb_hypertable": {"time_column_name": "timestamp"}},)


class TradeSignals(Base):
    __tablename__ = "trade_signals"
    __table_args__ = TIMESCALEDB_ARGS

    signal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    timestamp: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)

    account_name: Mapped[str] = mapped_column(String, nullable=False)
    token: Mapped[str] = mapped_column(String, nullable=False)
    strategic_directive: Mapped[str] = mapped_column(String, nullable=False)

    prediction_label: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_probability: Mapped[float] = mapped_column(Float, nullable=False)


class CurrentPositions(Base):
    __tablename__ = "current_positions"
    __table_args__ = (PrimaryKeyConstraint("account_name", "token"),)

    account_name: Mapped[str] = mapped_column(String, nullable=False)
    token: Mapped[str] = mapped_column(String, nullable=False)

    trade_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    direction: Mapped[str] = mapped_column(String, nullable=False)

    position_size: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    virtual_stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    virtual_take_profit: Mapped[float] = mapped_column(Float, nullable=False)

    entry_timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)


class TradeLogs(Base):
    __tablename__ = "trade_logs"
    __table_args__ = (
        {"timescaledb_hypertable": {"time_column_name": "entry_timestamp"}},
    )

    trade_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    signal_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    account_name: Mapped[str] = mapped_column(String, nullable=False)
    exchange: Mapped[str] = mapped_column(String, nullable=False)
    token: Mapped[str] = mapped_column(String, nullable=False)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)

    entry_exchange_order_id: Mapped[str | None] = mapped_column(String)
    exit_exchange_order_id: Mapped[str | None] = mapped_column(String)

    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Float)
    position_size: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_usd: Mapped[float | None] = mapped_column(Float)
    fees: Mapped[float | None] = mapped_column(Float)

    entry_timestamp: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    exit_timestamp: Mapped[int | None] = mapped_column(BigInteger, index=True)


__all__ = [
    "TradeSignals",
    "CurrentPositions",
    "TradeLogs",
]
