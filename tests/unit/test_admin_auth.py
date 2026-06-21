import pytest
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request

from app.admin.auth import AdminLoginAuth
from app.core.settings import get_settings
from app.main import app


def make_request(username: str, password: str) -> Request:
    """Реальный `starlette.requests.Request`, не мок — `slowapi` (ADM-4/ADM-5,
    `@limiter.limit` на `login()`) строго проверяет `isinstance(request, Request)`,
    мок (даже `spec=Request`) эту проверку не проходит."""
    scope = {
        'type': 'http', 'method': 'POST', 'path': '/admin/login', 'headers': [],
        'query_string': b'', 'client': ('127.0.0.1', 12345), 'app': app, 'session': {},
    }
    request = Request(scope)
    request._form = {'username': username, 'password': password}  # noqa: SLF001 — обходит парсинг multipart-тела
    return request


@pytest.mark.asyncio
async def test_login_succeeds_with_correct_credentials():
    settings = get_settings()
    auth = AdminLoginAuth(secret_key='test-secret')
    request = make_request(settings.app.admin_login, settings.app.admin_password.get_secret_value())

    result = await auth.login(request)

    assert result is True
    assert request.session['admin_authenticated'] is True


@pytest.mark.asyncio
async def test_login_fails_with_wrong_password():
    settings = get_settings()
    auth = AdminLoginAuth(secret_key='test-secret')
    request = make_request(settings.app.admin_login, 'definitely-wrong-password')

    result = await auth.login(request)

    assert result is False
    assert 'admin_authenticated' not in request.session


@pytest.mark.asyncio
async def test_login_fails_with_wrong_username():
    settings = get_settings()
    auth = AdminLoginAuth(secret_key='test-secret')
    request = make_request('not-the-admin', settings.app.admin_password.get_secret_value())

    result = await auth.login(request)

    assert result is False


def test_https_only_defaults_to_false_for_local_dev():
    auth = AdminLoginAuth(secret_key='test-secret')

    middleware = auth.middlewares[0]
    assert middleware.cls is SessionMiddleware
    assert middleware.kwargs['https_only'] is False


def test_https_only_true_is_propagated_to_session_middleware():
    auth = AdminLoginAuth(secret_key='test-secret', https_only=True)

    middleware = auth.middlewares[0]
    assert middleware.kwargs['https_only'] is True
