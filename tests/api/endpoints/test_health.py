from unittest.mock import AsyncMock

from httpx import AsyncClient

from app.dependencies.services import get_health_service
from app.exceptions.health import DatabaseUnavailableError
from app.main import app
from app.schemas.health import HealthSchema
from app.services.health import HealthService


async def test_get_health_returns_200_when_database_ok(async_client: AsyncClient):
    fake_service = AsyncMock(spec=HealthService)
    fake_service.check_health.return_value = HealthSchema(status='ok', database='ok')
    app.dependency_overrides[get_health_service] = lambda: fake_service

    response = await async_client.get('/api/v1/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'database': 'ok'}


async def test_get_health_returns_503_when_database_unavailable(async_client: AsyncClient):
    fake_service = AsyncMock(spec=HealthService)
    fake_service.check_health.side_effect = DatabaseUnavailableError(error_details='нет соединения')
    app.dependency_overrides[get_health_service] = lambda: fake_service

    response = await async_client.get('/api/v1/health')

    assert response.status_code == 503
    assert response.json()['detail'] == 'База данных недоступна.'
