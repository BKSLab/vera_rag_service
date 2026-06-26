from datetime import date
from unittest.mock import AsyncMock

from httpx import AsyncClient

from app.dependencies.services import get_documents_service, get_ingestion_service
from app.main import app
from app.models.schemas import SectionUpdateResponse
from app.services.documents import DocumentsService
from app.services.ingestion import IngestionService


async def test_delete_document_returns_200_with_document_id(async_client: AsyncClient):
    fake_service = AsyncMock(spec=DocumentsService)
    app.dependency_overrides[get_documents_service] = lambda: fake_service

    response = await async_client.delete('/api/v1/document/fz-181-art21')

    assert response.status_code == 200
    assert response.json() == {'document_id': 'fz-181-art21'}
    fake_service.delete_document.assert_awaited_once_with('fz-181-art21')


_SECTION_UPDATE_PAYLOAD = {
    'category': 'labor_code',
    'raw_text': 'Работник имеет право на ежегодный оплачиваемый отпуск продолжительностью 28 календарных дней.',
    'section_title': 'Статья 114. Ежегодные оплачиваемые отпуска',
    'version': '2026-01-01',
    'effective_date': '2026-01-01',
    'source_title': 'Трудовой кодекс Российской Федерации',
    'audience': 'both',
    'topic': 'leave',
}


async def test_update_section_returns_200_with_summary(async_client: AsyncClient):
    fake_service = AsyncMock(spec=IngestionService)
    fake_service.ingest_section.return_value = SectionUpdateResponse(
        document_id='tk-rf',
        section_number='114',
        parent_id='tk-rf:114',
        version='2026-01-01',
        chunks_count=2,
        superseded_chunks=1,
    )
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    response = await async_client.put('/api/v1/document/tk-rf/sections/114', json=_SECTION_UPDATE_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body['document_id'] == 'tk-rf'
    assert body['section_number'] == '114'
    assert body['parent_id'] == 'tk-rf:114'
    assert body['chunks_count'] == 2
    assert body['superseded_chunks'] == 1
    fake_service.ingest_section.assert_awaited_once()


async def test_update_section_returns_422_when_category_not_allowed(async_client: AsyncClient):
    fake_service = AsyncMock(spec=IngestionService)
    fake_service.ingest_section.side_effect = ValueError(
        "Гранулярное обновление не поддерживается для category='authorial'."
    )
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    payload = {**_SECTION_UPDATE_PAYLOAD, 'category': 'authorial'}
    response = await async_client.put('/api/v1/document/tk-rf/sections/114', json=payload)

    assert response.status_code == 422


async def test_update_section_returns_401_when_api_key_invalid(async_client: AsyncClient):
    from app.dependencies.auth import verify_api_key
    app.dependency_overrides.pop(verify_api_key, None)

    response = await async_client.put(
        '/api/v1/document/tk-rf/sections/114',
        json=_SECTION_UPDATE_PAYLOAD,
        headers={'X-API-Key': 'wrong-key'},
    )

    assert response.status_code == 401
