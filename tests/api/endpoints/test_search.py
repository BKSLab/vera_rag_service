from unittest.mock import AsyncMock

from httpx import AsyncClient

from app.dependencies.services import get_search_service
from app.exceptions.embedding import EmbeddingApiRequestError
from app.exceptions.llm import LlmApiRequestError
from app.main import app
from app.models.schemas import SearchResultChunk
from app.services.search import SearchService


async def test_search_chunks_returns_200_with_results(async_client: AsyncClient):
    fake_service = AsyncMock(spec=SearchService)
    fake_service.search.return_value = [
        SearchResultChunk(
            chunk_id='chunk-1',
            text='Текст чанка.',
            synthetic_title='Заголовок',
            source_title='ФЗ-181, Статья 21',
            audience='employer',
            topics=['quota'],
            category='federal_law',
            score=0.9,
        )
    ]
    app.dependency_overrides[get_search_service] = lambda: fake_service

    response = await async_client.post('/api/v1/search', json={'query': 'квота на инвалидов', 'top_k': 5})

    assert response.status_code == 200
    body = response.json()
    assert len(body['chunks']) == 1
    assert body['chunks'][0]['chunk_id'] == 'chunk-1'


async def test_search_chunks_returns_200_with_empty_list(async_client: AsyncClient):
    fake_service = AsyncMock(spec=SearchService)
    fake_service.search.return_value = []
    app.dependency_overrides[get_search_service] = lambda: fake_service

    response = await async_client.post('/api/v1/search', json={'query': 'нет совпадений'})

    assert response.status_code == 200
    assert response.json() == {'chunks': []}


async def test_search_chunks_returns_422_when_query_empty(async_client: AsyncClient):
    response = await async_client.post('/api/v1/search', json={'query': ''})

    assert response.status_code == 422


async def test_search_chunks_returns_500_when_embedding_api_fails(async_client: AsyncClient):
    fake_service = AsyncMock(spec=SearchService)
    fake_service.search.side_effect = EmbeddingApiRequestError(
        error_details='таймаут', request_url='https://embeddings.example'
    )
    app.dependency_overrides[get_search_service] = lambda: fake_service

    response = await async_client.post('/api/v1/search', json={'query': 'квота на инвалидов'})

    assert response.status_code == 500
    assert 'Embedding API' in response.json()['detail']


async def test_search_chunks_returns_500_when_reranker_llm_fails(async_client: AsyncClient):
    fake_service = AsyncMock(spec=SearchService)
    fake_service.search.side_effect = LlmApiRequestError(
        error_details='невалидный JSON', request_url='https://llm.example'
    )
    app.dependency_overrides[get_search_service] = lambda: fake_service

    response = await async_client.post('/api/v1/search', json={'query': 'квота на инвалидов'})

    assert response.status_code == 500
    assert 'LLM API' in response.json()['detail']
