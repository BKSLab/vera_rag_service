import pytest
from httpx import ASGITransport, AsyncClient

from app.dependencies.auth import verify_api_key
from app.main import app


@pytest.fixture(autouse=True)
def clear_overrides():
    """Снимает dependency_overrides после каждого теста, чтобы они не протекали в следующий."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def bypass_api_key_auth():
    """Большинство тестов проверяют поведение эндпоинта, не саму авторизацию —
    авторизация (ARCH-1/API-1/SEC-1) покрыта отдельно в `test_auth.py`, где
    эта зависимость намеренно не переопределяется."""
    app.dependency_overrides[verify_api_key] = lambda: None
    yield


@pytest.fixture
async def async_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        yield client
