import logging
from contextvars import ContextVar

# LOG-2 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — единый request_id
# генерируется один раз на HTTP-запрос (middleware, app/main.py), доступен и
# структурным логам (через RequestIdLogFilter), и SearchLog.request_id — без
# этого невозможно сопоставить строку лога с записью в search_logs по
# одному и тому же запросу.
_request_id_var: ContextVar[str | None] = ContextVar('request_id', default=None)


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)


def get_request_id() -> str | None:
    return _request_id_var.get()


class RequestIdLogFilter(logging.Filter):
    """Добавляет `request_id` в каждую запись лога — `-` вне контекста
    запроса (например, лог при старте приложения)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or '-'
        return True
