import pytest

from core.exceptions import (
    UnicornWealthError,
    APIError,
    WebSocketError,
    DatabaseError,
    FeatureCalculationError,
    TradeExecutionError,
)


def test_exception_hierarchy():
    assert issubclass(APIError, UnicornWealthError)
    assert issubclass(WebSocketError, UnicornWealthError)
    assert issubclass(DatabaseError, UnicornWealthError)
    assert issubclass(FeatureCalculationError, UnicornWealthError)
    assert issubclass(TradeExecutionError, UnicornWealthError)


def test_exceptions_can_be_raised():
    with pytest.raises(APIError):
        raise APIError("This is a test API error")

    with pytest.raises(WebSocketError):
        raise WebSocketError("This is a test WebSocket error")

    with pytest.raises(DatabaseError):
        raise DatabaseError("This is a test Database error")

    with pytest.raises(FeatureCalculationError):
        raise FeatureCalculationError("This is a test Feature Calculation error")

    with pytest.raises(TradeExecutionError):
        raise TradeExecutionError("This is a test Trade Execution error")
