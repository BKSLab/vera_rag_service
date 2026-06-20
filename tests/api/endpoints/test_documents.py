from unittest.mock import AsyncMock

from httpx import AsyncClient

from app.dependencies.services import get_documents_service
from app.main import app
from app.services.documents import DocumentsService


async def test_delete_document_returns_200_with_document_id(async_client: AsyncClient):
    fake_service = AsyncMock(spec=DocumentsService)
    app.dependency_overrides[get_documents_service] = lambda: fake_service

    response = await async_client.delete('/api/v1/document/fz-181-art21')

    assert response.status_code == 200
    assert response.json() == {'document_id': 'fz-181-art21'}
    fake_service.delete_document.assert_awaited_once_with('fz-181-art21')
