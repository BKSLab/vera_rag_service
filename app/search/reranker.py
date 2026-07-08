from dataclasses import dataclass

from app.clients.llm import LlmClient
from app.core.config_logger import logger
from app.exceptions.llm import LlmApiRequestError
from app.models.schemas import RerankResult
from app.search.prompts.reranker import RERANKER_PROMPT

RERANK_TOP_N = 5

# SEARCH-3 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — при
# категорийно-сбалансированном поиске кандидатов может быть до ~40
# (5 категорий × (top-4 dense + top-4 sparse)); без ограничения суммарная
# длина промпта растёт пропорционально размеру чанка и числу категорий без
# явного контроля.
#
# Исходное значение (600) исходило из предположения "начала чанка
# достаточно, чтобы оценить релевантность" — опровергнуто реальным поиском
# на полном ТК РФ (2026-07-08): запрос "увольнение за прогул инвалида"
# получил в кандидаты правильный чанк (статья 81, основания увольнения),
# но слово "прогул" в нём находится на 893-м символе — reranker обрезал
# текст до того, как до него дошёл, и вместо этого выбрал короткий, но
# менее релевантный чанк, целиком помещавшийся в лимит. Замер по всему
# индексу (765 чанков): p50=1169, p90=1934, p99=2014, max=2303 символа —
# **74% чанков длиннее 600** символов, то есть обрезка задевала
# подавляющее большинство кандидатов, а не только длинные редкие случаи.
# 2500 — с запасом покрывает full text практически всего корпуса.
CANDIDATE_TEXT_MAX_CHARS = 2500


@dataclass(frozen=True)
class RerankOutcome:
    """Результат reranker'а вместе со статусом для diagnostics/search_logs."""

    chunk_ids: list[str]
    status: str


def _build_candidates_prompt(query_text: str, candidates: list[tuple[str, str]]) -> str:
    """Оборачивает запрос и каждый кандидат в явные теги-разделители
    (LLM-3/SEC-5, AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — текст
    документов и пользовательский запрос — недоверенный вход, в котором
    теоретически может оказаться prompt injection. Теги сами по себе не
    защищают полностью (надёжной защиты не существует ни у одного
    провайдера), но дают модели чёткую границу "это данные" и снижают
    поверхность атаки в сочетании с инструкцией в `RERANKER_PROMPT`.

    Текст каждого кандидата урезается до `CANDIDATE_TEXT_MAX_CHARS` (SEARCH-3)
    — ограничивает суммарную длину промпта независимо от числа кандидатов
    и размера чанков.
    """
    numbered = '\n\n'.join(
        f'<candidate id="{i}">{text[:CANDIDATE_TEXT_MAX_CHARS]}</candidate>'
        for i, (_, text) in enumerate(candidates, start=1)
    )
    return f'<user_query>{query_text}</user_query>\n\nКандидаты:\n{numbered}'


def _map_indices_to_chunk_ids(
    ranked_indices: list[int], candidates: list[tuple[str, str]], top_n: int
) -> list[str]:
    """Валидирует номера от LLM (диапазон, дубликаты) и сопоставляет с chunk_id."""
    valid_indices = [index for index in ranked_indices if 1 <= index <= len(candidates)]
    deduped_indices = list(dict.fromkeys(valid_indices))[:top_n]
    return [candidates[index - 1][0] for index in deduped_indices]


async def rerank_chunks_with_status(
    llm_client: LlmClient,
    query_text: str,
    candidates: list[tuple[str, str]],
    top_n: int = RERANK_TOP_N,
) -> RerankOutcome:
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
        return RerankOutcome(chunk_ids=[], status='no_candidates')

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
        return RerankOutcome(
            chunk_ids=[chunk_id for chunk_id, _ in candidates[:top_n]],
            status='fallback_unavailable',
        )

    if result.ranked_indices == []:
        logger.info('ℹ️ Reranker: ни один кандидат не релевантен запросу, возвращается пустой список.')
        return RerankOutcome(chunk_ids=[], status='no_relevant')

    ranked_chunk_ids = _map_indices_to_chunk_ids(result.ranked_indices, candidates, top_n)
    if not ranked_chunk_ids:
        logger.warning('⚠️ Reranker вернул невалидные номера, используется исходный порядок RRF.')
        return RerankOutcome(
            chunk_ids=[chunk_id for chunk_id, _ in candidates[:top_n]],
            status='fallback_invalid_output',
        )

    return RerankOutcome(chunk_ids=ranked_chunk_ids, status='ok')


async def rerank_chunks(
    llm_client: LlmClient,
    query_text: str,
    candidates: list[tuple[str, str]],
    top_n: int = RERANK_TOP_N,
) -> list[str]:
    """Совместимая обёртка: возвращает только chunk_id без статуса."""
    outcome = await rerank_chunks_with_status(llm_client, query_text, candidates, top_n)
    return outcome.chunk_ids
