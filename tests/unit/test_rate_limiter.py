import asyncio
import time

from app.core.rate_limiter import RateLimiter


async def test_acquire_does_not_delay_first_call():
    limiter = RateLimiter(rate_per_second=10)

    started_at = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.05


async def test_acquire_enforces_minimum_interval_between_calls():
    limiter = RateLimiter(rate_per_second=10)  # min interval 0.1s

    await limiter.acquire()
    started_at = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - started_at

    assert elapsed >= 0.09  # с небольшим запасом на точность таймера


async def test_acquire_serializes_concurrent_callers_to_the_configured_rate():
    limiter = RateLimiter(rate_per_second=20)  # min interval 0.05s
    call_count = 5

    started_at = time.monotonic()
    await asyncio.gather(*(limiter.acquire() for _ in range(call_count)))
    elapsed = time.monotonic() - started_at

    # N вызовов "одновременно" всё равно растягиваются минимум на
    # (N-1) * min_interval — конкурентные вызовы не обходят лимитер.
    assert elapsed >= 0.05 * (call_count - 1) * 0.9
