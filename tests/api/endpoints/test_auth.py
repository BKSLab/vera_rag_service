from unittest.mock import AsyncMock

from httpx import AsyncClient

from app.core.settings import get_settings
from app.dependencies.auth import verify_api_key
from app.dependencies.services import get_health_service, get_search_service
from app.main import app
from app.schemas.health import HealthSchema
from app.services.health import HealthService
from app.services.search import SearchService


async def test_search_chunks_returns_422_when_api_key_header_missing(async_client: AsyncClient):
    app.dependency_overrides.pop(verify_api_key, None)

    response = await async_client.post('/api/v1/search', json={'query': 'квота на инвалидов'})

    assert response.status_code == 422


async def test_search_chunks_returns_401_when_api_key_invalid(async_client: AsyncClient):
    app.dependency_overrides.pop(verify_api_key, None)

    response = await async_client.post(
        '/api/v1/search',
        json={'query': 'квота на инвалидов'},
        headers={'X-API-Key': 'wrong-key'},
    )

    assert response.status_code == 401


async def test_search_chunks_returns_200_when_api_key_valid(async_client: AsyncClient):
    app.dependency_overrides.pop(verify_api_key, None)
    fake_service = AsyncMock(spec=SearchService)
    fake_service.search.return_value = []
    app.dependency_overrides[get_search_service] = lambda: fake_service

    response = await async_client.post(
        '/api/v1/search',
        json={'query': 'квота на инвалидов'},
        headers={'X-API-Key': get_settings().app.api_key.get_secret_value()},
    )

    assert response.status_code == 200


async def test_get_health_does_not_require_api_key(async_client: AsyncClient):
    app.dependency_overrides.pop(verify_api_key, None)
    fake_service = AsyncMock(spec=HealthService)
    fake_service.check_health.return_value = HealthSchema(status='ok', database='ok')
    app.dependency_overrides[get_health_service] = lambda: fake_service

    response = await async_client.get('/api/v1/health')

    assert response.status_code == 200
