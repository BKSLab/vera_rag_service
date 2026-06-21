import logging
import logging.config

from app.core.request_context import RequestIdLogFilter
from app.core.settings import get_settings

logging.config.fileConfig(
    get_settings().app.logging_config_path,
    disable_existing_loggers=False,
)

logger = logging.getLogger('vera_rag_service')

# LOG-2 — `request_id` в каждой записи лога этого логгера, не только в
# `search_logs`. Фильтр на уровне логгера, не отдельного хендлера — ловит
# любой хендлер, который добавят в будущем (например, отправку в систему
# агрегации, LOG-1), без необходимости помнить прицепить фильтр туда тоже.
logger.addFilter(RequestIdLogFilter())
