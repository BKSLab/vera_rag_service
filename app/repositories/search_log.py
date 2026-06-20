from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.search_log import SearchLog
from app.exceptions.search_log import SearchLogRepositoryError


class SearchLogRepository:
    """Запись журнала поисковых запросов (Этап 8). Только `save` — записи неизменяемы, без `update`/`delete`."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def save_search_log(self, search_log: SearchLog) -> None:
        """Сохраняет одну запись журнала поискового запроса.

        Args:
            search_log: Заполненная запись (без `id`/`created_at` — генерируются БД).

        Raises:
            SearchLogRepositoryError: При ошибке записи в БД.
        """
        try:
            self.db_session.add(search_log)
            await self.db_session.commit()
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise SearchLogRepositoryError(error_details=str(error)) from error
