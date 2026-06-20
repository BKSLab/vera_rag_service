from app.clients.llm import LlmClient
from app.core.config_logger import logger
from app.exceptions.llm import LlmApiRequestError
from app.models.schemas import RerankResult
from app.search.prompts.reranker import RERANKER_PROMPT

RERANK_TOP_N = 5


def _build_candidates_prompt(query_text: str, candidates: list[tuple[str, str]]) -> str:
    numbered = '\n\n'.join(f'[{i}] {text}' for i, (_, text) in enumerate(candidates, start=1))
    return f'Запрос пользователя: {query_text}\n\nКандидаты:\n{numbered}'


def _map_indices_to_chunk_ids(
    ranked_indices: list[int], candidates: list[tuple[str, str]], top_n: int
) -> list[str]:
    """Валидирует номера от LLM (диапазон, дубликаты) и сопоставляет с chunk_id."""
    valid_indices = [index for index in ranked_indices if 1 <= index <= len(candidates)]
    deduped_indices = list(dict.fromkeys(valid_indices))[:top_n]
    return [candidates[index - 1][0] for index in deduped_indices]


async def rerank_chunks(
    llm_client: LlmClient,
    query_text: str,
    candidates: list[tuple[str, str]],
    top_n: int = RERANK_TOP_N,
) -> list[str]:
    """Переранжирует top-20 кандидатов LLM'ом и возвращает top-N chunk_id (Этап 6).

    При отказе LLM (исчерпаны все retry) деградирует до исходного порядка
    кандидатов (RRF) вместо падения всего поискового запроса — reranker
    повышает качество, но его недоступность не должна блокировать поиск
    (см. FASTAPI_PATTERNS.md, раздел 9 — деградация при частичном отказе).

    Args:
        llm_client: LLM-клиент reranker'а (Polza/Gemini, не Yandex — см.
            `get_reranker_llm_client`).
        query_text: Текст запроса пользователя.
        candidates: Кандидаты в порядке RRF — (chunk_id, текст чанка).
        top_n: Сколько кандидатов вернуть после переранжирования.

    Returns:
        Список chunk_id, не длиннее top_n, в порядке убывания релевантности.
        Пустой список, если candidates пуст.
    """
    if not candidates:
        return []

    try:
        result: RerankResult = await llm_client.get_llm_response(
            content=_build_candidates_prompt(query_text, candidates),
            prompt=RERANKER_PROMPT,
            schema=RerankResult,
        )
    except LlmApiRequestError as error:
        logger.warning(
            '⚠️ Reranker недоступен, используется исходный порядок RRF. Детали: %s', error
        )
        return [chunk_id for chunk_id, _ in candidates[:top_n]]

    ranked_chunk_ids = _map_indices_to_chunk_ids(result.ranked_indices, candidates, top_n)
    if not ranked_chunk_ids:
        logger.warning('⚠️ Reranker не вернул валидных номеров, используется исходный порядок RRF.')
        return [chunk_id for chunk_id, _ in candidates[:top_n]]

    return ranked_chunk_ids
