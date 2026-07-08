from unittest.mock import AsyncMock

from httpx import AsyncClient

from app.dependencies.services import get_ingestion_service
from app.exceptions.embedding import EmbeddingApiRequestError
from app.exceptions.ingestion import RawTextTooLargeError, TooManyChunksError, TopicsNotAllowedForCategoryError
from app.exceptions.llm import LlmApiRequestError
from app.main import app
from app.models.schemas import MAX_RAW_TEXT_LENGTH, IngestResponse
from app.services.ingestion import IngestionService

VALID_PAYLOAD = {
    'document_id': 'fz-181-art21',
    'category': 'labor_code',
    'raw_text': 'Текст статьи 21 ФЗ-181.',
    'source_title': 'ФЗ-181, Статья 21',
    'audience': 'both',
    'topics': [],
    'version': '2026-01-01',
    'effective_date': '2026-01-01',
}


async def test_ingest_document_returns_200_with_summary(async_client: AsyncClient):
    fake_service = AsyncMock(spec=IngestionService)
    fake_service.ingest_document.return_value = IngestResponse(
        document_id='fz-181-art21', version='2026-01-01', chunks_count=3, replaced_versions=[]
    )
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    response = await async_client.post('/api/v1/ingest', json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body['chunks_count'] == 3
    assert body['replaced_versions'] == []


async def test_ingest_document_returns_422_when_category_invalid(async_client: AsyncClient):
    payload = {**VALID_PAYLOAD, 'category': 'video'}

    response = await async_client.post('/api/v1/ingest', json=payload)

    assert response.status_code == 422


async def test_ingest_document_returns_422_when_raw_text_exceeds_max_length(async_client: AsyncClient):
    payload = {**VALID_PAYLOAD, 'raw_text': 'a' * (MAX_RAW_TEXT_LENGTH + 1)}

    response = await async_client.post('/api/v1/ingest', json=payload)

    assert response.status_code == 422


async def test_ingest_document_returns_422_when_too_many_chunks(async_client: AsyncClient):
    fake_service = AsyncMock(spec=IngestionService)
    fake_service.ingest_document.side_effect = TooManyChunksError(
        document_id='fz-181-art21', chunks_count=2500, max_chunks=2000
    )
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    response = await async_client.post('/api/v1/ingest', json=VALID_PAYLOAD)

    assert response.status_code == 422


async def test_ingest_document_returns_422_when_raw_text_too_large_raised_by_service(async_client: AsyncClient):
    fake_service = AsyncMock(spec=IngestionService)
    fake_service.ingest_document.side_effect = RawTextTooLargeError(
        document_id='fz-181-art21', length=2_000_000, max_length=MAX_RAW_TEXT_LENGTH
    )
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    response = await async_client.post('/api/v1/ingest', json=VALID_PAYLOAD)

    assert response.status_code == 422


async def test_ingest_document_returns_422_when_topics_not_allowed_for_category(async_client: AsyncClient):
    fake_service = AsyncMock(spec=IngestionService)
    fake_service.ingest_document.side_effect = TopicsNotAllowedForCategoryError(
        document_id='fz-181-art21', category='federal_law', topics=['квотирование'],
    )
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    response = await async_client.post('/api/v1/ingest', json={**VALID_PAYLOAD, 'topics': ['квотирование']})

    assert response.status_code == 422


async def test_ingest_document_returns_500_when_llm_enrichment_fails(async_client: AsyncClient):
    fake_service = AsyncMock(spec=IngestionService)
    fake_service.ingest_document.side_effect = LlmApiRequestError(
        error_details='LLM недоступен', request_url='https://llm.example'
    )
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    response = await async_client.post('/api/v1/ingest', json=VALID_PAYLOAD)

    assert response.status_code == 500
    assert 'LLM API' in response.json()['detail']


async def test_ingest_document_returns_500_when_embedding_fails(async_client: AsyncClient):
    fake_service = AsyncMock(spec=IngestionService)
    fake_service.ingest_document.side_effect = EmbeddingApiRequestError(
        error_details='Embedding недоступен', request_url='https://embeddings.example'
    )
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    response = await async_client.post('/api/v1/ingest', json=VALID_PAYLOAD)

    assert response.status_code == 500
    assert 'Embedding API' in response.json()['detail']
