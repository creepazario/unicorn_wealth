"""Lightweight pybreaker shim for test environments.

This module emulates a small subset of the pybreaker API used in the codebase,
so that tests can run even when the real dependency is not installed.

If the real `pybreaker` is installed, base_client will import it first and this
shim will be ignored.
"""

from __future__ import annotations

from typing import Callable

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half-open"


class CircuitBreakerError(Exception):
    pass


class CircuitBreaker:
    def __init__(self, fail_max: int = 5, reset_timeout: int = 60) -> None:
        self.fail_max = max(int(fail_max), 1)
        self.reset_timeout = int(reset_timeout)
        self._fail_counter = 0
        self._state = STATE_CLOSED

    # For compatibility with code checking .state or .is_open()
    @property
    def state(self) -> str:
        return self._state

    def is_open(self) -> bool:  # pragma: no cover - trivial
        return self._state == STATE_OPEN

    def _on_success(self) -> None:
        self._fail_counter = 0
        if self._state != STATE_CLOSED:
            self._state = STATE_CLOSED

    def _on_failure(self) -> None:
        self._fail_counter += 1
        if self._fail_counter >= self.fail_max:
            self._state = STATE_OPEN

    def call(self, func: Callable, *args, **kwargs):
        # If open, raise immediately per expected behavior
        if self._state == STATE_OPEN:
            raise CircuitBreakerError("Circuit is open")
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:  # noqa: BLE001 - propagate after recording
            self._on_failure()
            raise
