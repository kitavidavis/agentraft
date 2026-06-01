import pytest

from agentraft import BreakerState, CircuitBreaker, CircuitBreakerOpen, RetryPolicy


def test_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=100)
    for _ in range(2):
        cb.record_failure()
    assert cb.state is BreakerState.CLOSED
    cb.record_failure()  # third failure trips it
    assert cb.state is BreakerState.OPEN
    with pytest.raises(CircuitBreakerOpen):
        cb.check()


def test_breaker_resets_on_success():
    cb = CircuitBreaker(failure_threshold=2)
    cb.record_failure()
    cb.record_success()
    assert cb.consecutive_failures == 0
    assert cb.state is BreakerState.CLOSED


def test_breaker_half_open_after_cooldown():
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.0)
    cb.record_failure()
    # cooldown of 0 means it should immediately be half-open on next read
    assert cb.state is BreakerState.HALF_OPEN
    cb.check()  # half-open permits a trial


def test_retry_policy_backoff():
    rp = RetryPolicy(max_retries=3, base_delay=1.0, max_delay=8.0)
    assert rp.delay_for(0) == 0.0
    assert rp.delay_for(1) == 1.0
    assert rp.delay_for(2) == 2.0
    assert rp.delay_for(3) == 4.0
    assert rp.delay_for(10) == 8.0  # capped


def test_retry_policy_no_delay_by_default():
    rp = RetryPolicy(max_retries=2)
    assert rp.delay_for(1) == 0.0
