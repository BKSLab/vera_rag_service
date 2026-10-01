from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db.models.ingestion_run import IngestionRun


class IngestionRunRepository:
    def __init__(self, session_factory: async_sessionmaker):
        self.session_factory = session_factory

    async def save(self, values: dict) -> None:
        # Каждая запись видна сразу, в том числе пока основная обработка ждёт
        # блокировку или упала и откатывает свою транзакцию.
        statement = insert(IngestionRun).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[IngestionRun.id],
            set_={key: getattr(statement.excluded, key) for key in values if key != 'id'},
        )
        async with self.session_factory() as session, session.begin():
            await session.execute(statement)
