import asyncio
import time

# Обнаружено на реальной загрузке ТК РФ (2026-07-08): `EMBEDDING_CONCURRENCY`
# (asyncio.Semaphore) ограничивает конкурентность — сколько запросов "в
# полёте" одновременно — а квота Yandex Embedding API (`HTTP 429
# "ai.embeddingsTextEmbeddingRequestsPerSecond.rate ... allowed 10
# requests"`) — это ограничение по ЧАСТОТЕ (запросов/сек), не по
# конкурентности. Семафор не защищает от превышения: если единичный запрос
# быстрый, даже concurrency=3 успевает выдать десятки запросов в секунду —
# как только один из 3 "слотов" освобождается, тут же уходит следующий
# запрос. При достаточно большом документе (сотни чанков × до 6 запросов
# каждый) это не просто шумные логи — риск исчерпать все retry на каком-то
# чанке и уронить весь `ingest_document` (ING-4, "всё или ничего").
#
# Самописный простой leaky-bucket, не отдельная библиотека — требование
# единственное (не больше N запросов/сек на один процесс), не обосновывает
# внешнюю зависимость. Один экземпляр на провайдера — как и `CircuitBreaker`
# (`app/core/circuit_breaker.py`), должен быть module-level singleton в
# `app/dependencies/clients.py`, а не создаваться заново на каждый `EmbeddingClient`
# (тот создаётся через `Depends()` заново на каждый HTTP-запрос) — иначе
# каждый экземпляр считал бы свою частоту независимо, не деля общую квоту.


class RateLimiter:
    """Ограничивает частоту вызовов `acquire()` — не более `rate_per_second`
    раз в секунду суммарно по всем вызывающим (не на вызывающего). Вызовы,
    успевающие раньше минимального интервала, дожидаются своей очереди —
    без сброса лишних попыток, как это делает retry/circuit breaker."""

    def __init__(self, rate_per_second: float):
        self._min_interval = 1.0 / rate_per_second
        self._lock = asyncio.Lock()
        self._last_call_at: float | None = None

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._last_call_at is not None:
                wait_seconds = self._last_call_at + self._min_interval - now
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
            self._last_call_at = max(now, (self._last_call_at or 0) + self._min_interval)
