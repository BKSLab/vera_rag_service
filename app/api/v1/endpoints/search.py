from fastapi import APIRouter, HTTPException, Request, status

from app.core.config_logger import logger
from app.core.rate_limit import limiter
from app.dependencies.auth import VerifyApiKeyDep
from app.dependencies.services import SearchServiceDep
from app.exceptions.embedding import EmbeddingApiRequestError
from app.exceptions.llm import LlmApiRequestError
from app.models.schemas import SearchFilters, SearchRequest, SearchResponse

router = APIRouter(dependencies=[VerifyApiKeyDep])


@router.post(
    path='/search',
    status_code=status.HTTP_200_OK,
    summary='Семантический поиск по базе знаний',
    description=(
        'Гибридный поиск (dense + BM25 + RRF) с переранжированием LLM '
        '(Этапы 5–6 плана). Контракт для MCP Tools Server (раздел 5 плана).'
    ),
    operation_id='searchChunks',
    response_description='Найденные чанки, отсортированные по релевантности.',
    responses={
        200: {
            'description': 'Поиск выполнен (возможен пустой список chunks).',
            'content': {'application/json': {'example': {'chunks': []}}},
        },
        500: {
            'description': 'Ошибка при запросе к Embedding API или LLM-reranker.',
            'content': {'application/json': {'example': {'detail': 'Ошибка при запросе к Embedding API. Подробности: ...'}}},
        },
    },
    response_model=SearchResponse,
)
@limiter.limit('60/minute')
async def search_chunks(request: Request, data: SearchRequest, service: SearchServiceDep) -> SearchResponse:
    """Выполняет семантический поиск по базе знаний.

    Args:
        request: HTTP-запрос (нужен `slowapi` для определения IP клиента, API-2/SEC-2).
        data: Запрос — текст + опциональные фильтры по метаданным.
        service: Сервис поиска.

    Returns:
        Чанки, отсортированные по релевантности.
    """
    # LOG-4/SEC-8 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — текст
    # запроса не логируется в stdout: тематика сервиса (права людей с
    # инвалидностью) делает реалистичным присутствие в запросах сведений о
    # здоровье/инвалидности — специальная категория персональных данных по
    # 152-ФЗ. Полный текст остаётся только в `search_logs` (Postgres,
    # доступ — через защищённую авторизацией админку), не дублируется в
    # стандартные логи, которые обычно живут в системах со слабее
    # контролируемым доступом, чем основная БД.
    logger.info('🚀 Запрос POST /search. Длина запроса: %d символов.', len(data.query))
    try:
        filters = SearchFilters(audience=data.audience, topic=data.topic, category=data.category)
        chunks = await service.search(query=data.query, filters=filters, top_k=data.top_k)
        logger.info('✅ Запрос POST /search выполнен. Найдено: %d.', len(chunks))
        return SearchResponse(chunks=chunks)
    except (EmbeddingApiRequestError, LlmApiRequestError) as error:
        logger.exception('❌ Ошибка при поиске. Детали: %s', error)
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error
