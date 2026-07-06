from dataclasses import dataclass

from app.clients.llm import LlmClient
from app.core.config_logger import logger
from app.exceptions.llm import LlmApiRequestError
from app.models.schemas import QueryExpansionResult
from app.search.prompts.query_expansion import QUERY_EXPANSION_PROMPT


@dataclass(frozen=True)
class QueryExpansionOutcome:
    """Результат расширения запроса вместе со статусом fallback."""

    queries: list[str]
    status: str


def _flatten_variants(result: QueryExpansionResult, original_query: str) -> list[str]:
    """Разворачивает подвопросы и их переформулировки в плоский список
    текстов запроса, без дублей, с сохранением порядка появления.

    Падение к исходному запросу, если LLM вернул только пустые строки
    (теоретически возможно после `cap_rephrasings`/`cap_variants`, хотя
    схема запрещает пустой `variants`/`sub_question`).
    """
    queries = [
        text
        for variant in result.variants
        for text in (variant.sub_question, *variant.rephrasings)
        if text.strip()
    ]
    deduped = list(dict.fromkeys(queries))
    return deduped or [original_query]


async def expand_query_with_status(llm_client: LlmClient, query_text: str) -> QueryExpansionOutcome:
    """Декомпозирует составной запрос на подвопросы и переформулирует
    каждый ближе к терминологии трудового права (раздел 8 плана).

    Один LLM-вызов перед `hybrid_search`: для простого запроса — до
    `MAX_REPHRASINGS_PER_SUB_QUESTION` юридических переформулировок этого
    же смысла (повышение recall за счёт лексического разнообразия), для
    составного — декомпозиция на независимые подвопросы (не более
    `MAX_SUB_QUESTIONS`), каждый из которых также переформулируется.
    Каждый вариант — отдельный `hybrid_search`, результаты сливаются через
    RRF (см. `SearchService`).

    При отказе LLM (исчерпаны все retry) деградирует до одного варианта —
    исходного запроса без изменений, по аналогии с reranker'ом (Этап 6):
    расширение запроса повышает recall, но его недоступность не должна
    блокировать сам поиск.

    Args:
        llm_client: LLM-клиент расширения запроса (Polza/Gemini).
        query_text: Исходный текст запроса пользователя.

    Returns:
        Список текстов запроса для параллельного `hybrid_search`, без
        дублей. Всегда содержит хотя бы исходный запрос.
    """
    try:
        result: QueryExpansionResult = await llm_client.get_llm_response(
            content=f'<user_query>{query_text}</user_query>',
            prompt=QUERY_EXPANSION_PROMPT,
            schema=QueryExpansionResult,
        )
    except LlmApiRequestError as error:
        logger.warning(
            '⚠️ Расширение запроса недоступно, используется только исходный запрос. Детали: %s', error
        )
        return QueryExpansionOutcome(queries=[query_text], status='fallback_unavailable')

    return QueryExpansionOutcome(queries=_flatten_variants(result, query_text), status='ok')


async def expand_query(llm_client: LlmClient, query_text: str) -> list[str]:
    """Совместимая обёртка: возвращает только варианты запроса без статуса."""
    outcome = await expand_query_with_status(llm_client, query_text)
    return outcome.queries
