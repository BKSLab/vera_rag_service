from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from starlette.responses import Response

from app.core.settings import get_settings


class AdminLoginAuth(AuthenticationBackend):
    """Логин/пароль для входа в админку — отдельная плоскость доступа от
    API-ключей сервиса (FASTAPI_PATTERNS.md, раздел 6, «Двухуровневая авторизация»):
    `admin_login`/`admin_password` сверяются напрямую с `Settings`, без БД."""

    async def login(self, request: Request) -> bool:
        settings = get_settings()
        form = await request.form()
        username = form.get('username', '')
        password = form.get('password', '')
        if username == settings.app.admin_login and password == settings.app.admin_password.get_secret_value():
            request.session['admin_authenticated'] = True
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return request.session.get('admin_authenticated', False)
