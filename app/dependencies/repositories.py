from typing import Annotated

from fastapi import Depends

from app.dependencies.db_session import DbSessionDep
from app.repositories.document import DocumentRepository
from app.repositories.search_log import SearchLogRepository


def get_search_log_repository(db_session: DbSessionDep) -> SearchLogRepository:
    return SearchLogRepository(db_session)


SearchLogRepositoryDep = Annotated[SearchLogRepository, Depends(get_search_log_repository)]


def get_document_repository(db_session: DbSessionDep) -> DocumentRepository:
    return DocumentRepository(db_session)


DocumentRepositoryDep = Annotated[DocumentRepository, Depends(get_document_repository)]
