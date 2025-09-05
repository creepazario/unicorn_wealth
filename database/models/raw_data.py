"""Raw data ORM models for external API ingestions.

This module defines TimescaleDB-backed tables for storing raw, unprocessed
market and social datasets. Each table is configured as a hypertable on the
`timestamp` column to optimize time-series operations.
"""

from __future__ import annotations

from sqlalchemy.types import BigInteger, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


TIMESCALEDB_ARGS = ({"timescaledb_hypertable": {"time_column_name": "timestamp"}},)


class RawOHLCV(Base):
    """Raw OHLCV candles as received from providers.

    Columns:
        - id: surrogate primary key
        - timestamp: UNIX epoch milliseconds/seconds (BigInteger), indexed
        - token: asset identifier/symbol, indexed
        - interval: candle interval (e.g., 1m, 5m, 1h)
        - open, high, low, close, volume: float values from source
    """

    __tablename__ = "raw_ohlcv"
    __table_args__ = TIMESCALEDB_ARGS

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    token: Mapped[str] = mapped_column(String, index=True, nullable=False)
    interval: Mapped[str] = mapped_column(String, nullable=False)

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)


class RawFundingRates(Base):
    """Raw perpetual funding rates from derivatives venues."""

    __tablename__ = "raw_funding_rates"
    __table_args__ = TIMESCALEDB_ARGS

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    token: Mapped[str] = mapped_column(String, index=True, nullable=False)

    rate: Mapped[float] = mapped_column(Float, nullable=False)


class RawSocialMetrics(Base):
    """Raw social metrics as provided by third-party APIs."""

    __tablename__ = "raw_social_metrics"
    __table_args__ = TIMESCALEDB_ARGS

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    token: Mapped[str] = mapped_column(String, index=True, nullable=False)

    social_dominance: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment: Mapped[float] = mapped_column(Float, nullable=False)


__all__ = [
    "RawOHLCV",
    "RawFundingRates",
    "RawSocialMetrics",
]
