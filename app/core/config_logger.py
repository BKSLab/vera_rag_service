import logging
import logging.config

from app.core.request_context import RequestIdLogFilter
from app.core.settings import get_settings

logging.config.fileConfig(
    get_settings().app.logging_config_path,
    disable_existing_loggers=False,
)

# Фильтр на уровне хендлера — request_id добавляется в record до форматирования
# для ВСЕХ логгеров (httpx, hypercorn, vera_rag_service), не только нашего.
_request_id_filter = RequestIdLogFilter()
for _handler in logging.root.handlers:
    _handler.addFilter(_request_id_filter)

logger = logging.getLogger('vera_rag_service')
