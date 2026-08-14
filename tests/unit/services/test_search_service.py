from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import app.services.search as search_service_module
from app.clients.embeddings import EmbeddingClient
from app.clients.llm import LlmClient
from app.exceptions.search_log import SearchLogRepositoryError
from app.models.schemas import QueryExpansionResult, QueryVariant, RerankResult, SearchFilters
from app.repositories.search_log import SearchLogRepository
from app.services.search import SearchService

CHUNK_ID = '11111111-1111-1111-1111-111111111111'
CHUNK_PAYLOAD = {
    'text': 'Квота на трудоустройство инвалидов — 2%.',
    'synthetic_title': 'Квота',
    'source_title': 'ФЗ-181, Статья 21',
    'audience': 'employer',
    'topics': ['quota'],
    'category': 'federal_law',
}


def _build_service(*, has_candidates: bool = True) -> tuple[SearchService, AsyncMock]:
    embedding_client = AsyncMock(spec=EmbeddingClient)
    embedding_client.get_embedding.return_value = [0.1, 0.2, 0.3]

    qdrant_client = AsyncMock()
    if has_candidates:
        qdrant_client.query_points.return_value = SimpleNamespace(
            points=[SimpleNamespace(id=CHUNK_ID, score=0.9)]
        )
        qdrant_client.scroll.return_value = ([SimpleNamespace(id=CHUNK_ID, payload=CHUNK_PAYLOAD)], None)
        qdrant_client.retrieve.return_value = [SimpleNamespace(id=CHUNK_ID, payload=CHUNK_PAYLOAD)]
    else:
        qdrant_client.query_points.return_value = SimpleNamespace(points=[])
        qdrant_client.scroll.return_value = ([], None)

    vector_store = SimpleNamespace(client=qdrant_client, collection_name='vera_kb')

    reranker_llm_client = AsyncMock(spec=LlmClient)
    reranker_llm_client.get_llm_response.return_value = RerankResult(ranked_indices=[1])

    query_expansion_llm_client = AsyncMock(spec=LlmClient)
    query_expansion_llm_client.get_llm_response.return_value = QueryExpansionResult(
        variants=[QueryVariant(sub_question='квота на инвалидов', rephrasings=[])]
    )

    search_log_repository = AsyncMock(spec=SearchLogRepository)

    service = SearchService(
        embedding_client=embedding_client,
        reranker_llm_client=reranker_llm_client,
        query_expansion_llm_client=query_expansion_llm_client,
        vector_store=vector_store,
        search_log_repository=search_log_repository,
    )
    return service, search_log_repository


async def test_search_returns_reranked_chunk_and_saves_log():
    service, search_log_repository = _build_service()

    results = await service.search(query='квота на инвалидов', filters=SearchFilters(audience='employer'), top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == CHUNK_ID
    assert results[0].rerank_rank == 1
    search_log_repository.save_search_log.assert_awaited_once()
    saved_log = search_log_repository.save_search_log.await_args.args[0]
    assert saved_log.query == 'квота на инвалидов'
    assert saved_log.audience == 'employer'
    assert saved_log.reranked_chunk_ids == [CHUNK_ID]
    assert saved_log.query_expansion_status == 'ok'
    assert saved_log.reranker_status == 'ok'
    assert len(saved_log.final_response) == 1
    assert saved_log.latency_embed_query_ms >= 0
    assert saved_log.latency_hybrid_search_ms >= 0
    assert saved_log.latency_rerank_ms >= 0
    reranker_prompt = service.reranker_llm_client.get_llm_response.await_args.kwargs['content']
    assert 'source_title=ФЗ-181, Статья 21' in reranker_prompt
    assert 'category=federal_law' in reranker_prompt


async def test_search_returns_empty_list_and_saves_log_when_no_candidates():
    service, search_log_repository = _build_service(has_candidates=False)

    results = await service.search(query='нет совпадений', filters=SearchFilters(), top_k=5)

    assert results == []
    search_log_repository.save_search_log.assert_awaited_once()
    saved_log = search_log_repository.save_search_log.await_args.args[0]
    assert saved_log.rrf_candidates == []
    assert saved_log.final_response == []
    assert saved_log.reranker_status == 'no_candidates'


async def test_search_degrades_when_search_log_save_fails():
    service, search_log_repository = _build_service()
    search_log_repository.save_search_log.side_effect = SearchLogRepositoryError(error_details='нет соединения с БД')

    results = await service.search(query='квота на инвалидов', filters=SearchFilters(), top_k=5)

    assert len(results) == 1


async def test_search_top_k_ten_returns_ten_chunks_and_logs_candidate_sources(monkeypatch):
    service, _ = _build_service()
    chunk_ids = [f'{index:08d}-1111-1111-1111-111111111111' for index in range(10)]
    points = [SimpleNamespace(id=chunk_id, score=1.0 - index / 100) for index, chunk_id in enumerate(chunk_ids)]
    service.vector_store.client.query_points.return_value = SimpleNamespace(points=points)
    service.vector_store.client.retrieve.return_value = [
        SimpleNamespace(id=chunk_id, payload={**CHUNK_PAYLOAD, 'text': f'Текст чанка {index}'})
        for index, chunk_id in enumerate(chunk_ids)
    ]
    service.reranker_llm_client.get_llm_response.return_value = RerankResult(
        ranked_indices=list(range(1, 11))
    )
    log_info = Mock()
    monkeypatch.setattr(search_service_module.logger, 'info', log_info)

    results = await service.search(query='квота на инвалидов', filters=SearchFilters(), top_k=10)

    assert [result.chunk_id for result in results] == chunk_ids
    assert [result.rerank_rank for result in results] == list(range(1, 11))
    reranker_prompt = service.reranker_llm_client.get_llm_response.await_args.kwargs['prompt']
    assert 'не более 10' in reranker_prompt
    source_log_calls = [call for call in log_info.call_args_list if 'Top-кандидат' in call.args[0]]
    assert len(source_log_calls) == 10
    assert all('chunk-dense:' in call.args[2] for call in source_log_calls)
    assert all('question-dense:' in call.args[2] for call in source_log_calls)
    assert all('sparse:' in call.args[2] for call in source_log_calls)
