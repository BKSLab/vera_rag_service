import hmac

from sqladmin.authentication import AuthenticationBackend
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request

from app.core.rate_limit import limiter
from app.core.settings import get_settings


class AdminLoginAuth(AuthenticationBackend):
    """Логин/пароль для входа в админку — отдельная плоскость доступа от
    API-ключей сервиса (FASTAPI_PATTERNS.md, раздел 6, «Двухуровневая авторизация»):
    `admin_login`/`admin_password` сверяются напрямую с `Settings`, без БД."""

    def __init__(self, secret_key: str, https_only: bool = False):
        super().__init__(secret_key=secret_key)
        # ADM-7 — явный `https_only` для cookie сессии вместо дефолтного
        # `SessionMiddleware` без него (sqladmin сам не выставляет этот флаг).
        self.middlewares = [Middleware(SessionMiddleware, secret_key=secret_key, https_only=https_only)]

    @limiter.limit('5/minute')
    async def login(self, request: Request) -> bool:
        settings = get_settings()
        form = await request.form()
        username = form.get('username', '')
        password = form.get('password', '')
        # ADM-4/ADM-5 — `hmac.compare_digest` вместо `==`: сравнение секретов
        # обычным `==` не constant-time, теоретическая поверхность для
        # тайминг-атаки на единственную учётную запись с полным доступом к БЗ.
        username_valid = hmac.compare_digest(username, settings.app.admin_login)
        password_valid = hmac.compare_digest(password, settings.app.admin_password.get_secret_value())
        if username_valid and password_valid:
            request.session['admin_authenticated'] = True
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return request.session.get('admin_authenticated', False)
