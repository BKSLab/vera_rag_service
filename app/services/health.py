from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.health import DatabaseUnavailableError
from app.schemas.health import HealthSchema


class HealthService:
    """Проверка статуса сервиса и критичных внешних зависимостей."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def check_health(self) -> HealthSchema:
        """Проверяет доступность Postgres и возвращает агрегированный статус.

        Returns:
            Статус сервиса и зависимостей.

        Raises:
            DatabaseUnavailableError: Если БД недоступна.
        """
        try:
            await self.db_session.execute(text('SELECT 1'))
        except SQLAlchemyError as error:
            raise DatabaseUnavailableError(error_details=str(error)) from error

        return HealthSchema(status='ok', database='ok')
