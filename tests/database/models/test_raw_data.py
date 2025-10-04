import pytest

from database.models.raw_data import (
    RawOHLCV15m,
    RawOHLCV1h,
    RawOHLCV4h,
    RawOHLCV1d,
    RawOHLCV7d,
    RawFundingRates,
    RawSocialMetrics,
)


_ = pytest  # silence flake8 unused warning


def _assert_ohlcv_columns(model_cls):
    # Common columns for OHLCV tables
    for col in ("timestamp", "token", "open", "high", "low", "close", "volume"):
        assert hasattr(model_cls, col)


def test_raw_ohlcv_models_exist_with_expected_names():
    assert RawOHLCV15m.__tablename__ == "raw_ohlcv_15m"
    assert RawOHLCV1h.__tablename__ == "raw_ohlcv_1h"
    assert RawOHLCV4h.__tablename__ == "raw_ohlcv_4h"
    assert RawOHLCV1d.__tablename__ == "raw_ohlcv_1d"
    assert RawOHLCV7d.__tablename__ == "raw_ohlcv_7d"

    # Validate expected columns on each model
    for model in (RawOHLCV15m, RawOHLCV1h, RawOHLCV4h, RawOHLCV1d, RawOHLCV7d):
        _assert_ohlcv_columns(model)


def _assert_common_time_series_attrs(model_cls):
    # Common columns
    for col in ("id", "timestamp", "token"):
        assert hasattr(model_cls, col)


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
