from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.admin import create_admin
from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.ingest import router as ingest_router
from app.api.v1.endpoints.search import router as search_router
from app.core.config_logger import logger
from app.db.session import engine
from app.dependencies.vectorstore import get_vector_store
from app.vectorstore.client import qdrant_client


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


app = FastAPI(title='vera_rag_service', lifespan=lifespan)

app.mount('/static', StaticFiles(directory=Path(__file__).parent / 'static'), name='static')
create_admin(app=app, engine=engine)

app.include_router(health_router, prefix='/api/v1')
app.include_router(search_router, prefix='/api/v1')
app.include_router(ingest_router, prefix='/api/v1')
app.include_router(documents_router, prefix='/api/v1')
