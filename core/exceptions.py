"""Custom exception hierarchy for the UnicornWealth application.

This module defines a base exception and several specific exception types to
enable precise and predictable error handling across the codebase.
"""


class UnicornWealthError(Exception):
    """Base exception for all UnicornWealth-specific errors."""

    pass


class APIError(UnicornWealthError):
    """Errors related to API interactions."""

    pass


class WebSocketError(UnicornWealthError):
    """Errors related to WebSocket connections and messaging."""

    pass


class DatabaseError(UnicornWealthError):
    """Errors related to database operations."""

    pass


class FeatureCalculationError(UnicornWealthError):
    """Errors occurring during feature engineering/calculation."""

    pass


class TradeExecutionError(UnicornWealthError):
    """Errors occurring while executing trades."""

    pass
