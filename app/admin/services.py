from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.clients.http_client import external_api_http_client
from app.db.session import async_session_factory
from app.dependencies.clients import (
    get_embedding_client,
    get_enrichment_llm_client,
    get_query_expansion_llm_client,
    get_reranker_llm_client,
)
from app.dependencies.vectorstore import get_vector_store
from app.repositories.document import DocumentRepository
from app.repositories.search_log import SearchLogRepository
from app.services.documents import DocumentsService
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
    # ARCH-6 — общий `external_api_http_client`, не новый `httpx.AsyncClient`
    # на каждый вызов формы загрузки.
    async with async_session_factory() as db_session:
        yield IngestionService(
            llm_client=get_enrichment_llm_client(external_api_http_client),
            embedding_client=get_embedding_client(external_api_http_client),
            vector_store=get_vector_store(),
            document_repository=DocumentRepository(db_session),
        )


@asynccontextmanager
async def build_search_service() -> AsyncIterator[SearchService]:
    async with async_session_factory() as db_session:
        yield SearchService(
            embedding_client=get_embedding_client(external_api_http_client),
            reranker_llm_client=get_reranker_llm_client(external_api_http_client),
            query_expansion_llm_client=get_query_expansion_llm_client(external_api_http_client),
            vector_store=get_vector_store(),
            search_log_repository=SearchLogRepository(db_session),
        )


@asynccontextmanager
async def build_documents_service() -> AsyncIterator[DocumentsService]:
    """Используется `DocumentAdmin.delete_model` (ARCH-4) — единая точка
    удаления документа, общая с публичным `DELETE /document/{id}`."""
    async with async_session_factory() as db_session:
        yield DocumentsService(
            vector_store=get_vector_store(),
            document_repository=DocumentRepository(db_session),
        )
