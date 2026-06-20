from typing import Annotated

from fastapi import Depends

from app.dependencies.db_session import DbSessionDep
from app.repositories.search_log import SearchLogRepository


def get_search_log_repository(db_session: DbSessionDep) -> SearchLogRepository:
    return SearchLogRepository(db_session)


SearchLogRepositoryDep = Annotated[SearchLogRepository, Depends(get_search_log_repository)]
