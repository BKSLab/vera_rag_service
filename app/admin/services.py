from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from app.db.session import async_session_factory
from app.dependencies.clients import get_embedding_client, get_llm_client, get_reranker_llm_client
from app.dependencies.vectorstore import get_vector_store
from app.repositories.document import DocumentRepository
from app.repositories.search_log import SearchLogRepository
from app.services.ingestion import IngestionService
from app.services.search import SearchService

# sqladmin BaseView-страницы (Этап 11 плана) не проходят через FastAPI
# Depends() — это плоскость доступа над Starlette Request, без контейнера
# DI эндпоинтов. Дублировать саму бизнес-логику ingestion/поиска здесь
# нельзя (раздел 11.2 плана: "без дублирования логики"), поэтому собираем
# те же сервисы вручную из тех же чистых функций-зависимостей
# (`app/dependencies/*.py`), что и FastAPI — отличается только способ
# вызова, не логика.


@asynccontextmanager
async def build_ingestion_service() -> AsyncIterator[IngestionService]:
    async with httpx.AsyncClient() as httpx_client, async_session_factory() as db_session:
        yield IngestionService(
            llm_client=get_llm_client(httpx_client),
            embedding_client=get_embedding_client(httpx_client),
            vector_store=get_vector_store(),
            document_repository=DocumentRepository(db_session),
        )


@asynccontextmanager
async def build_search_service() -> AsyncIterator[SearchService]:
    async with httpx.AsyncClient() as httpx_client, async_session_factory() as db_session:
        yield SearchService(
            embedding_client=get_embedding_client(httpx_client),
            reranker_llm_client=get_reranker_llm_client(httpx_client),
            vector_store=get_vector_store(),
            search_log_repository=SearchLogRepository(db_session),
        )
