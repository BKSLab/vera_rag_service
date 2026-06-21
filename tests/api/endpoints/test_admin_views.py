import pytest
from httpx import AsyncClient

from app.core.rate_limit import limiter
from app.core.settings import get_settings

ADMIN_BASE_VIEW_PATHS = [
    '/admin/dashboard',
    '/admin/document-upload',
    '/admin/document-chunks',
    '/admin/search-test',
]
ADMIN_MODEL_VIEW_LIST_PATHS = [
    '/admin/document/list',
    '/admin/search-log/list',
]


@pytest.fixture(autouse=True)
def reset_limiter():
    """`/admin/login` лимитирован (`5/minute`, ADM-4/ADM-5) — общее in-memory
    состояние `Limiter` на весь тестовый прогон, не только этот файл."""
    limiter.reset()
    yield
    limiter.reset()


@pytest.mark.parametrize('path', ADMIN_BASE_VIEW_PATHS)
async def test_base_view_redirects_unauthenticated_request_to_login(async_client: AsyncClient, path: str):
    """Регрессия на конкретный, уже однажды найденный и исправленный баг
    (RAG_SERVICE_PLAN.md, Этап 11.1): `Admin.add_base_view()`, вызванный
    напрямую, не проставляет `_admin_ref`, из-за чего `login_required`
    тихо пропускает проверку авторизации и страница отвечает 200 без
    логина вместо редиректа. TEST-4
    (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md)."""
    response = await async_client.get(path)

    assert response.status_code in (302, 303)
    assert '/admin/login' in response.headers['location']


@pytest.mark.parametrize('path', ADMIN_MODEL_VIEW_LIST_PATHS)
async def test_model_view_redirects_unauthenticated_request_to_login(async_client: AsyncClient, path: str):
    response = await async_client.get(path)

    assert response.status_code in (302, 303)
    assert '/admin/login' in response.headers['location']


async def test_login_with_valid_credentials_then_dashboard_is_accessible(async_client: AsyncClient):
    settings = get_settings().app
    login_response = await async_client.post(
        '/admin/login',
        data={'username': settings.admin_login, 'password': settings.admin_password.get_secret_value()},
    )
    assert login_response.status_code in (302, 303)

    dashboard_response = await async_client.get('/admin/dashboard')

    assert dashboard_response.status_code == 200


async def test_login_with_invalid_credentials_does_not_grant_access(async_client: AsyncClient):
    login_response = await async_client.post(
        '/admin/login', data={'username': 'not-admin', 'password': 'wrong-password'},
    )
    assert login_response.status_code == 400  # sqladmin: форма логина с ошибкой, не редирект на успех

    dashboard_response = await async_client.get('/admin/dashboard')

    assert dashboard_response.status_code in (302, 303)
