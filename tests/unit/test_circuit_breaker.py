import time

from app.core.circuit_breaker import CircuitBreaker


def test_circuit_breaker_starts_closed():
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)

    assert breaker.is_open() is False


def test_circuit_breaker_opens_after_threshold_consecutive_failures():
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open() is False

    breaker.record_failure()
    assert breaker.is_open() is True


def test_circuit_breaker_closes_on_success():
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=10.0)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open() is True

    breaker.record_success()
    assert breaker.is_open() is False


def test_circuit_breaker_closes_automatically_after_reset_timeout():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)

    breaker.record_failure()
    assert breaker.is_open() is True

    time.sleep(0.1)
    assert breaker.is_open() is False
