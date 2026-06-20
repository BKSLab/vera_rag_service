from typing import Annotated

from fastapi import Depends

from app.dependencies.clients import EmbeddingClientDep, LlmClientDep, RerankerLlmClientDep
from app.dependencies.db_session import DbSessionDep
from app.dependencies.repositories import SearchLogRepositoryDep
from app.dependencies.vectorstore import VectorStoreDep
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
    vector_store: VectorStoreDep,
    search_log_repository: SearchLogRepositoryDep,
) -> SearchService:
    return SearchService(
        embedding_client=embedding_client,
        reranker_llm_client=reranker_llm_client,
        vector_store=vector_store,
        search_log_repository=search_log_repository,
    )


SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]


def get_ingestion_service(
    llm_client: LlmClientDep,
    embedding_client: EmbeddingClientDep,
    vector_store: VectorStoreDep,
) -> IngestionService:
    return IngestionService(
        llm_client=llm_client, embedding_client=embedding_client, vector_store=vector_store
    )


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]


def get_documents_service(vector_store: VectorStoreDep) -> DocumentsService:
    return DocumentsService(vector_store=vector_store)


DocumentsServiceDep = Annotated[DocumentsService, Depends(get_documents_service)]
