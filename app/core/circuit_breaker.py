import time

# LLM-2 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — без circuit breaker
# каждый независимый HTTP-запрос к сервису самостоятельно проходит полный
# цикл retry с экспоненциальным backoff, прежде чем вернуть ошибку, даже
# если предыдущие N запросов уже показали, что провайдер недоступен. При
# продолжительном внешнем инциденте это означает синхронный рост latency
# всех запросов вместо быстрого отказа.
#
# Самописный, не `aiobreaker`/`purgatory` — требования модальные (N
# последовательных отказов → открыт на T секунд), отдельная зависимость не
# обоснована для такого объёма логики. Один экземпляр на провайдера+use-case
# (не на клиент!) — должен переживать конкретный HTTP-запрос, в рамках
# которого создаётся клиент через `Depends()`, поэтому создаётся как
# module-level singleton в `app/dependencies/clients.py`, не внутри
# `LlmClient`/`EmbeddingClient`.


class CircuitBreaker:
    """Простой circuit breaker: открывается после N последовательных
    отказов, закрывается автоматически через `reset_timeout` секунд
    (half-open — следующий вызов после таймаута допускается, его результат
    определяет, остаться ли открытым)."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_timeout:
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


class CircuitBreakerOpenError(Exception):
    """Поднимается вместо реального HTTP-вызова, когда breaker открыт —
    провайдер уже показал `failure_threshold` подряд отказов недавно."""

    def __init__(self, reset_timeout: float):
        self.reset_timeout = reset_timeout
        super().__init__(f'Circuit breaker открыт — провайдер недоступен, повтор через ~{reset_timeout:.0f}с.')
