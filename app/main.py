from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.admin import create_admin
from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.ingest import router as ingest_router
from app.api.v1.endpoints.search import router as search_router
from app.clients.http_client import external_api_http_client
from app.core.config_logger import logger
from app.core.rate_limit import limiter
from app.core.request_context import set_request_id
from app.db.session import engine
from app.dependencies.vectorstore import get_vector_store
from app.vectorstore.client import qdrant_client

REQUEST_ID_HEADER = 'X-Request-ID'


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('🚀 Проверка подключения к Postgres при старте сервиса.')
    try:
        async with engine.connect() as connection:
            await connection.execute(text('SELECT 1'))
    except SQLAlchemyError:
        logger.exception('❌ Не удалось подключиться к Postgres при старте.')
        raise
    logger.info('✅ Подключение к Postgres подтверждено.')

    logger.info('🚀 Проверка коллекции Qdrant при старте сервиса.')
    await get_vector_store().ensure_collection()
    logger.info('✅ Коллекция Qdrant готова.')

    yield

    await engine.dispose()
    await qdrant_client.close()
    await external_api_http_client.aclose()


app = FastAPI(title='vera_rag_service', lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.middleware('http')
async def request_id_middleware(request: Request, call_next):
    """LOG-2 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — единый
    `request_id` на запрос, доступный и структурным логам (через
    `RequestIdLogFilter`), и `SearchLog.request_id` — раньше каждый
    генерировался отдельно и независимо, что не позволяло сопоставить
    строку лога с записью в `search_logs` по одному и тому же запросу.
    Принимает уже заданный клиентом `X-Request-ID` (для трассировки через
    несколько сервисов — MCP Tools Server может передать свой), иначе
    генерирует новый.
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
    set_request_id(request_id)
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response

app.mount('/static', StaticFiles(directory=Path(__file__).parent / 'static'), name='static')
create_admin(app=app, engine=engine)

app.include_router(health_router, prefix='/api/v1')
app.include_router(search_router, prefix='/api/v1')
app.include_router(ingest_router, prefix='/api/v1')
app.include_router(documents_router, prefix='/api/v1')

# LOG-5 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — минимальные
# метрики (latency/error rate по эндпоинту) без полного distributed tracing
# (раздел "Этап 8" RAG_SERVICE_PLAN.md осознанно отказался от OTel/Phoenix
# по другой причине — межсервисная трассировка, не базовые метрики самого
# сервиса). `/metrics` — без авторизации (Prometheus-скрейперы обычно не
# поддерживают произвольные заголовки) — в production ограничивать доступ
# на уровне сети/firewall, не пытаться защитить тем же `X-API-Key`.
Instrumentator().instrument(app).expose(app, endpoint='/metrics', include_in_schema=False)
