from unittest.mock import AsyncMock

from app.clients.llm import LlmClient
from app.exceptions.llm import LlmApiRequestError
from app.models.schemas import RerankResult
from app.search.reranker import (
    CANDIDATE_TEXT_MAX_CHARS,
    _build_candidates_prompt,
    rerank_chunks,
    rerank_chunks_with_status,
)


def make_candidates(count: int = 5) -> list[tuple[str, str]]:
    return [(f'chunk-{i}', f'Текст кандидата {i}') for i in range(count)]


async def test_rerank_chunks_returns_empty_list_for_no_candidates():
    llm_client = AsyncMock(spec=LlmClient)

    result = await rerank_chunks(llm_client, 'запрос', [])

    assert result == []
    llm_client.get_llm_response.assert_not_called()


async def test_rerank_chunks_maps_ranked_indices_to_chunk_ids():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = RerankResult(ranked_indices=[3, 1, 5])
    candidates = make_candidates(5)

    result = await rerank_chunks(llm_client, 'запрос', candidates)

    assert result == ['chunk-2', 'chunk-0', 'chunk-4']


async def test_rerank_chunks_drops_out_of_range_and_duplicate_indices():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = RerankResult(ranked_indices=[1, 99, 1, 2, 0])
    candidates = make_candidates(3)

    result = await rerank_chunks(llm_client, 'запрос', candidates)

    assert result == ['chunk-0', 'chunk-1']


async def test_rerank_chunks_respects_top_n():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = RerankResult(ranked_indices=[1, 2, 3, 4, 5])
    candidates = make_candidates(5)

    result = await rerank_chunks(llm_client, 'запрос', candidates, top_n=2)

    assert result == ['chunk-0', 'chunk-1']


async def test_rerank_chunks_falls_back_to_original_order_when_llm_unavailable():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.side_effect = LlmApiRequestError(error_details='boom', request_url='https://x')
    candidates = make_candidates(7)

    result = await rerank_chunks(llm_client, 'запрос', candidates, top_n=5)

    assert result == ['chunk-0', 'chunk-1', 'chunk-2', 'chunk-3', 'chunk-4']


async def test_rerank_chunks_with_status_reports_unavailable_fallback():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.side_effect = LlmApiRequestError(error_details='boom', request_url='https://x')
    candidates = make_candidates(3)

    result = await rerank_chunks_with_status(llm_client, 'запрос', candidates, top_n=2)

    assert result.chunk_ids == ['chunk-0', 'chunk-1']
    assert result.status == 'fallback_unavailable'


async def test_rerank_chunks_falls_back_when_llm_returns_no_valid_indices():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = RerankResult(ranked_indices=[99, 100])
    candidates = make_candidates(3)

    result = await rerank_chunks(llm_client, 'запрос', candidates, top_n=2)

    assert result == ['chunk-0', 'chunk-1']


async def test_rerank_chunks_with_status_reports_invalid_output_fallback():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = RerankResult(ranked_indices=[99, 100])
    candidates = make_candidates(3)

    result = await rerank_chunks_with_status(llm_client, 'запрос', candidates, top_n=2)

    assert result.chunk_ids == ['chunk-0', 'chunk-1']
    assert result.status == 'fallback_invalid_output'


async def test_rerank_chunks_returns_empty_list_when_llm_signals_no_relevant_candidates():
    """Этап 6.1 — LLM вернул ranked_indices=[], что означает 'ничего релевантного'.
    Это НЕ деградация к RRF, а чистый сигнал для Agent Service."""
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = RerankResult(ranked_indices=[])
    candidates = make_candidates(5)

    result = await rerank_chunks(llm_client, 'вопрос не по теме', candidates)

    assert result == []


async def test_rerank_chunks_with_status_reports_no_relevant_candidates():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = RerankResult(ranked_indices=[])
    candidates = make_candidates(5)

    result = await rerank_chunks_with_status(llm_client, 'вопрос не по теме', candidates)

    assert result.chunk_ids == []
    assert result.status == 'no_relevant'


def test_build_candidates_prompt_truncates_long_candidate_text():
    """SEARCH-3 — суммарная длина промпта не должна расти неограниченно
    с размером чанка/числом категорий."""
    candidates = [('chunk-0', 'а' * (CANDIDATE_TEXT_MAX_CHARS + 500))]

    prompt = _build_candidates_prompt('запрос', candidates)

    assert 'а' * (CANDIDATE_TEXT_MAX_CHARS + 1) not in prompt
    assert 'а' * CANDIDATE_TEXT_MAX_CHARS in prompt
