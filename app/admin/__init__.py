from pathlib import Path

from fastapi import FastAPI
from sqladmin import Admin
from sqlalchemy.ext.asyncio import AsyncEngine

from app.admin.auth import AdminLoginAuth
from app.admin.views import DocumentAdmin, DocumentUploadView, SearchLogAdmin, SearchTestView
from app.core.settings import get_settings

_TEMPLATES_DIR = str(Path(__file__).parent.parent / 'templates')


def create_admin(app: FastAPI, engine: AsyncEngine) -> Admin:
    """Единая фабрика админки (раздел 14 FASTAPI_PATTERNS.md). Стиль и подход
    (тёмная тема, sqladmin, login/password как отдельная плоскость доступа
    от API-ключей) — общие для всех сервисов продукта «Работа для всех»,
    взяты из api_work_for_everyone."""
    settings = get_settings()

    admin = Admin(
        app=app,
        engine=engine,
        authentication_backend=AdminLoginAuth(secret_key=settings.app.secret_key.get_secret_value()),
        title='Vera RAG Service — Admin',
        base_url='/admin',
        templates_dir=_TEMPLATES_DIR,
    )
    admin.add_view(SearchLogAdmin)
    admin.add_view(DocumentAdmin)
    # `add_view` (не `add_base_view` напрямую!) — только она проставляет
    # `_admin_ref` на класс view'а, без которого `login_required` (sqladmin)
    # тихо пропускает проверку авторизации для @expose-маршрутов BaseView.
    admin.add_view(DocumentUploadView)
    admin.add_view(SearchTestView)

    return admin
