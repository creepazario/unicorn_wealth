import pytest

from unicorn_wealth.database.models.raw_data import (
    RawOHLCV,
    RawFundingRates,
    RawSocialMetrics,
)


_ = pytest  # silence flake8 unused warning


def _assert_common_time_series_attrs(model_cls):
    # __table_args__ should exist and be non-empty (TimescaleDB config present)
    assert hasattr(model_cls, "__table_args__")
    args = getattr(model_cls, "__table_args__")
    assert args is not None
    # Defined in models as a tuple containing a dict of timescaledb options
    assert isinstance(args, tuple)
    assert len(args) > 0

    # Common columns
    assert hasattr(model_cls, "id")
    assert hasattr(model_cls, "timestamp")
    assert hasattr(model_cls, "token")


def test_raw_ohlcv_model():
    # Table name
    assert RawOHLCV.__tablename__ == "raw_ohlcv"

    _assert_common_time_series_attrs(RawOHLCV)

    # Specific columns
    for col in ("interval", "open", "high", "low", "close", "volume"):
        assert hasattr(RawOHLCV, col)


def test_raw_funding_rates_model():
    # Table name
    assert RawFundingRates.__tablename__ == "raw_funding_rates"

    _assert_common_time_series_attrs(RawFundingRates)

    # Specific columns
    assert hasattr(RawFundingRates, "rate")


def test_raw_social_metrics_model():
    # Table name
    assert RawSocialMetrics.__tablename__ == "raw_social_metrics"

    _assert_common_time_series_attrs(RawSocialMetrics)

    # Specific columns
    assert hasattr(RawSocialMetrics, "social_dominance")
    assert hasattr(RawSocialMetrics, "sentiment")
