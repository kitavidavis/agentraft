"""Circuit breaker + retry policy.

The circuit breaker prevents an error cascade (and runaway token cost) when a
pipeline keeps failing verification. After ``failure_threshold`` consecutive
failures it *opens* and rejects further attempts until a cooldown elapses, at
which point it goes *half-open* and allows a single trial.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from .errors import CircuitBreakerOpen


class BreakerState(str, Enum):
    CLOSED = "closed"        # normal operation
    OPEN = "open"            # tripped — rejecting attempts
    HALF_OPEN = "half_open"  # trial attempt allowed after cooldown


class CircuitBreaker:
    """Consecutive-failure circuit breaker with a cooldown."""

    def __init__(self, *, failure_threshold: int = 5, cooldown_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> BreakerState:
        # Auto-transition OPEN -> HALF_OPEN once the cooldown has elapsed.
        if self._state is BreakerState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                self._state = BreakerState.HALF_OPEN
        return self._state

    def check(self) -> None:
        """Raise :class:`CircuitBreakerOpen` if no attempt is currently permitted."""
        if self.state is BreakerState.OPEN:
            remaining = self.cooldown_seconds - (time.monotonic() - (self._opened_at or 0))
            raise CircuitBreakerOpen(
                f"Circuit breaker open after {self._consecutive_failures} consecutive "
                f"failures; retry in {max(0, remaining):.1f}s."
            )

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._state = BreakerState.OPEN
            self._opened_at = time.monotonic()
        elif self._state is BreakerState.HALF_OPEN:
            # Trial failed — re-open immediately.
            self._state = BreakerState.OPEN
            self._opened_at = time.monotonic()

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures


class RetryPolicy:
    """Per-step retry budget with exponential backoff."""

    def __init__(self, *, max_retries: int = 2, base_delay: float = 0.0, max_delay: float = 8.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def delay_for(self, attempt: int) -> float:
        """Backoff delay before the given (0-based) attempt."""
        if self.base_delay <= 0 or attempt <= 0:
            return 0.0
        return min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
