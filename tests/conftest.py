import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

from app.db.models.base import Base


@pytest_asyncio.fixture(scope='session')
def postgres_container():
    """Один контейнер Postgres на весь тестовый прогон — поднимать его на каждый тест слишком дорого."""
    with PostgresContainer('postgres:16-alpine') as container:
        yield container


@pytest_asyncio.fixture
async def engine(postgres_container):
    """Function-scoped (не session, как в эталонном примере FASTAPI_PATTERNS.md):
    pytest-asyncio 0.25 даёт каждому тесту свой event loop, и engine, созданный в
    session-scoped fixture на одном loop, ловит `RuntimeError: ... different loop`
    при использовании в тесте на другом. Контейнер Postgres всё равно один на
    весь прогон (session-scoped `postgres_container`) — пересоздаётся только
    легковесный `AsyncEngine`, не сама БД."""
    url = postgres_container.get_connection_url().replace('psycopg2', 'asyncpg')
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(engine):
    """Сессия на отдельный тест.

    `rollback()` после теста не освобождает от изоляции сам по себе — репозитории
    (`SearchLogRepository`/`DocumentRepository`) вызывают `commit()` внутри своих
    методов, и `rollback()` откатывает только то, что произошло после последнего
    commit (то есть ничего). Поэтому после rollback дополнительно очищаем все
    таблицы — иначе тесты в одном прогоне делят закоммиченные строки друг друга
    (см. AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md, TEST-1).
    """
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()
    async with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())
