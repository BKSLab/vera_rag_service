import hmac
import secrets

from starlette.requests import Request

# ADM-3 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — формы админки
# (`DocumentUploadView`/`SearchTestView`) запускают реальные действия
# (платный ingestion, поисковый запрос) без CSRF-токена. Synchronizer
# token pattern — токен живёт в сессии (уже подписанной/зашифрованной
# `SessionMiddleware`, sqladmin), не нужна отдельная подпись через
# `itsdangerous` для самого токена.
CSRF_SESSION_KEY = 'csrf_token'


def get_or_create_csrf_token(request: Request) -> str:
    """Возвращает CSRF-токен текущей сессии, создавая его при первом GET."""
    token = request.session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        request.session[CSRF_SESSION_KEY] = token
    return token


def verify_csrf_token(request: Request, submitted_token: str | None) -> bool:
    """Сверяет токен из тела формы с тем, что лежит в сессии (`hmac.compare_digest`)."""
    expected = request.session.get(CSRF_SESSION_KEY)
    if not expected or not submitted_token:
        return False
    return hmac.compare_digest(submitted_token, expected)
