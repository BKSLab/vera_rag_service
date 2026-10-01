from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import async_sessionmaker
from starlette.requests import Request

from app.dependencies.clients import (
    EmbeddingClientDep,
    EnrichmentLlmClientDep,
    LegalSyncClientDep,
    QueryExpansionLlmClientDep,
    RerankerLlmClientDep,
)
from app.dependencies.db_session import DbSessionDep
from app.dependencies.repositories import (
    DocumentChangeLogRepositoryDep,
    DocumentRepositoryDep,
    SearchLogRepositoryDep,
)
from app.dependencies.vectorstore import VectorStoreDep
from app.repositories.ingestion_run import IngestionRunRepository
from app.services.documents import DocumentsService
from app.services.health import HealthService
from app.services.ingestion import IngestionService
from app.services.search import SearchService


def get_health_service(db_session: DbSessionDep) -> HealthService:
    return HealthService(db_session=db_session)


HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]


def get_search_service(
    embedding_client: EmbeddingClientDep,
    reranker_llm_client: RerankerLlmClientDep,
    query_expansion_llm_client: QueryExpansionLlmClientDep,
    vector_store: VectorStoreDep,
    search_log_repository: SearchLogRepositoryDep,
) -> SearchService:
    return SearchService(
        embedding_client=embedding_client,
        reranker_llm_client=reranker_llm_client,
        query_expansion_llm_client=query_expansion_llm_client,
        vector_store=vector_store,
        search_log_repository=search_log_repository,
    )


SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]


def get_ingestion_service(
    request: Request,
    llm_client: EnrichmentLlmClientDep,
    embedding_client: EmbeddingClientDep,
    vector_store: VectorStoreDep,
    document_repository: DocumentRepositoryDep,
    change_log_repository: DocumentChangeLogRepositoryDep,
    legal_sync_client: LegalSyncClientDep,
) -> IngestionService:
    return IngestionService(
        llm_client=llm_client,
        embedding_client=embedding_client,
        vector_store=vector_store,
        document_repository=document_repository,
        change_log_repository=change_log_repository,
        legal_sync_client=legal_sync_client,
        ingestion_run_repository=IngestionRunRepository(async_sessionmaker(document_repository.db_session.bind, expire_on_commit=False)),
        source='legal_sync' if _change_id(request) else 'api',
        change_id=_change_id(request),
    )


def _change_id(request: Request) -> int | None:
    value = request.headers.get('X-Legal-Sync-Change-ID', '')
    return int(value) if len(value) <= 10 and value.isascii() and value.isdecimal() and 0 < int(value) < 2**31 else None


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]


def get_documents_service(
    vector_store: VectorStoreDep, document_repository: DocumentRepositoryDep
) -> DocumentsService:
    return DocumentsService(vector_store=vector_store, document_repository=document_repository)


DocumentsServiceDep = Annotated[DocumentsService, Depends(get_documents_service)]
