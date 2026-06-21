from unittest.mock import AsyncMock

from httpx import AsyncClient

from app.dependencies.services import get_health_service
from app.main import app
from app.schemas.health import HealthSchema
from app.services.health import HealthService


def _mock_health_service() -> None:
    fake_service = AsyncMock(spec=HealthService)
    fake_service.check_health.return_value = HealthSchema(status='ok', database='ok')
    app.dependency_overrides[get_health_service] = lambda: fake_service


async def test_response_includes_generated_request_id_header(async_client: AsyncClient):
    """LOG-2 — request_id генерируется на запрос и виден в ответе."""
    _mock_health_service()

    response = await async_client.get('/api/v1/health')

    assert 'X-Request-ID' in response.headers
    assert len(response.headers['X-Request-ID']) > 0


async def test_response_echoes_client_supplied_request_id(async_client: AsyncClient):
    """LOG-2 — если вызывающий (например, MCP Tools Server) уже передал свой
    request_id, сервис его не подменяет — нужно для трассировки через
    несколько сервисов одним и тем же идентификатором."""
    _mock_health_service()

    response = await async_client.get('/api/v1/health', headers={'X-Request-ID': 'caller-supplied-id'})

    assert response.headers['X-Request-ID'] == 'caller-supplied-id'
