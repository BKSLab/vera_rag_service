from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from app.core.rate_limit import limiter
from app.dependencies.services import get_search_service
from app.main import app
from app.services.search import SearchService


@pytest.fixture(autouse=True)
def reset_limiter():
    """`Limiter` хранит состояние в памяти процесса — общее на весь тестовый
    прогон, не только на этот файл. Сброс до и после, чтобы не съедать
    лимит других тестов и не унести собственное превышение в следующий тест."""
    limiter.reset()
    yield
    limiter.reset()


async def test_search_returns_429_after_exceeding_rate_limit(async_client: AsyncClient):
    """API-2/SEC-2 — лимит реально срабатывает, не просто синтаксически подключён."""
    fake_service = AsyncMock(spec=SearchService)
    fake_service.search.return_value = []
    app.dependency_overrides[get_search_service] = lambda: fake_service

    responses = [
        await async_client.post('/api/v1/search', json={'query': 'квота на инвалидов'}) for _ in range(61)
    ]

    assert responses[-1].status_code == 429
