"""Raw data ORM models for external API ingestions.

This module defines TimescaleDB-backed tables for storing raw, unprocessed
market and social datasets. Each table is configured as a hypertable on the
`timestamp` column to optimize time-series operations.
"""

from __future__ import annotations

from .base import Base
from sqlalchemy import Column, Integer, String, Float, BigInteger, UniqueConstraint

TIMESCALEDB_ARGS = ({"timescaledb_hypertable": {"time_column_name": "timestamp"}},)


class RawOHLCV15m(Base):
    __tablename__ = "raw_ohlcv_15m"
    timestamp = Column(BigInteger, primary_key=True)
    token = Column(String, primary_key=True)
    open = Column(Float(53), nullable=False)
    high = Column(Float(53), nullable=False)
    low = Column(Float(53), nullable=False)
    close = Column(Float(53), nullable=False)
    volume = Column(Float(53), nullable=False)
    __table_args__ = (
        UniqueConstraint("timestamp", "token", name="_timestamp_token_uc_15m"),
    )


class RawOHLCV1h(Base):
    __tablename__ = "raw_ohlcv_1h"
    timestamp = Column(BigInteger, primary_key=True)
    token = Column(String, primary_key=True)
    open = Column(Float(53), nullable=False)
    high = Column(Float(53), nullable=False)
    low = Column(Float(53), nullable=False)
    close = Column(Float(53), nullable=False)
    volume = Column(Float(53), nullable=False)
    __table_args__ = (
        UniqueConstraint("timestamp", "token", name="_timestamp_token_uc_1h"),
    )


class RawOHLCV4h(Base):
    __tablename__ = "raw_ohlcv_4h"
    timestamp = Column(BigInteger, primary_key=True)
    token = Column(String, primary_key=True)
    open = Column(Float(53), nullable=False)
    high = Column(Float(53), nullable=False)
    low = Column(Float(53), nullable=False)
    close = Column(Float(53), nullable=False)
    volume = Column(Float(53), nullable=False)
    __table_args__ = (
        UniqueConstraint("timestamp", "token", name="_timestamp_token_uc_4h"),
    )


class RawOHLCV1d(Base):
    __tablename__ = "raw_ohlcv_1d"
    timestamp = Column(BigInteger, primary_key=True)
    token = Column(String, primary_key=True)
    open = Column(Float(53), nullable=False)
    high = Column(Float(53), nullable=False)
    low = Column(Float(53), nullable=False)
    close = Column(Float(53), nullable=False)
    volume = Column(Float(53), nullable=False)
    __table_args__ = (
        UniqueConstraint("timestamp", "token", name="_timestamp_token_uc_1d"),
    )


class RawFundingRates(Base):
    """Raw perpetual funding rates from derivatives venues."""

    __tablename__ = "raw_funding_rates"
    __table_args__ = TIMESCALEDB_ARGS

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(BigInteger, index=True, nullable=False)
    token = Column(String, index=True, nullable=False)

    rate = Column(Float, nullable=False)


class RawSocialMetrics(Base):
    """Raw social metrics as provided by third-party APIs."""

    __tablename__ = "raw_social_metrics"
    __table_args__ = TIMESCALEDB_ARGS

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(BigInteger, index=True, nullable=False)
    token = Column(String, index=True, nullable=False)

    social_dominance = Column(Float, nullable=False)
    sentiment = Column(Float, nullable=False)


class RawOHLCV7d(Base):
    __tablename__ = "raw_ohlcv_7d"
    timestamp = Column(BigInteger, primary_key=True)
    token = Column(String, primary_key=True)
    open = Column(Float(53), nullable=False)
    high = Column(Float(53), nullable=False)
    low = Column(Float(53), nullable=False)
    close = Column(Float(53), nullable=False)
    volume = Column(Float(53), nullable=False)
    __table_args__ = (
        UniqueConstraint("timestamp", "token", name="_timestamp_token_uc_7d"),
    )


__all__ = [
    "RawOHLCV15m",
    "RawOHLCV1h",
    "RawOHLCV4h",
    "RawOHLCV1d",
    "RawOHLCV7d",
    "RawFundingRates",
    "RawSocialMetrics",
]
