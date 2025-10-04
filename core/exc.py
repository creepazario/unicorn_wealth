from __future__ import annotations


class ExchangeError(Exception):
    """Base exception for exchange-related errors."""


class CircuitBreakerOpenError(ExchangeError):
    """Raised when the circuit breaker is open and calls are not allowed."""

    def __init__(
        self, message: str = "Circuit breaker is open; operation blocked."
    ) -> None:
        super().__init__(message)
