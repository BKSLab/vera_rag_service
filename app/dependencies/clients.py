from typing import Annotated

from fastapi import Depends

from app.clients.embeddings import EmbeddingClient
from app.clients.llm import LlmClient
from app.core.circuit_breaker import CircuitBreaker
from app.core.rate_limiter import RateLimiter
from app.core.settings import get_settings
from app.dependencies.http_client import HttpClientDep

# LLM-2 (RAG_SERVICE_PLAN.md, раздел 7) — один breaker на
# провайдера+use-case, не на запрос: должен переживать конкретный
# `LlmClient`/`EmbeddingClient`, который DI создаёт заново на каждый
# HTTP-запрос. Независимые breaker'ы на обогащение и reranker — отказ
# одного не должен "размыкать" вызовы другого, даже если оба теперь у
# одного провайдера (Polza).
_yandex_embedding_breaker = CircuitBreaker()
_polza_reranker_breaker = CircuitBreaker()
_polza_enrichment_breaker = CircuitBreaker()
_polza_query_expansion_breaker = CircuitBreaker()

# Реальный лимит Yandex Embedding API — 10 запросов/сек на аккаунт (узнали
# из тела ответа 429, "allowed 10 requests"), общий и на ingestion
# (doc-модель), и на query-time эмбеддинг поиска (query-модель) — отсюда
# module-level singleton, а не поле EmbeddingClient (тот DI создаёт заново
# на каждый HTTP-запрос, независимый rate limiter на каждый экземпляр не
# делил бы общую квоту). 8, не 10 — запас на неточность таймера и на то,
# что ingestion и поиск могут одновременно расходовать одну и ту же квоту.
_yandex_embedding_rate_limiter = RateLimiter(rate_per_second=8)


def get_enrichment_llm_client(httpx_client: HttpClientDep) -> LlmClient:
    """LLM-клиент для обогащения чанков (Этап 3) — Polza AI (Gemini), не Yandex.

    Перешли с прямого Yandex Cloud API (`yandexgpt`/`yandexgpt-lite`) на
    Polza/Gemini 2026-06-21: на ~2% статей ТК РФ (мобилизация, военная
    служба, религиозные организации) Yandex Cloud отказывался отвечать
    («Я не могу обсуждать эту тему») на уровне платформенной модерации —
    не лечится ни промптом, ни сменой модели Yandex (проверено и на Pro,
    и на Lite, оба отказали на тех же фрагментах). Текст обогащения —
    нормативный акт РФ (публичный, не персональные данные), поэтому
    требование 152-ФЗ о резидентности данных (раздел 0.1 плана) к этому
    вызову не применяется — оно касалось обработки запросов
    пользователей, не обогащения корпуса базы знаний при загрузке.
    """
    settings = get_settings().polza
    return LlmClient(
        httpx_client=httpx_client,
        model=settings.polza_enrichment_llm_model,
        url=settings.polza_llm_api_url,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {settings.polza_api_key.get_secret_value()}',
        },
        timeout=settings.polza_enrichment_timeout_seconds,
        retries=settings.polza_enrichment_retries,
        circuit_breaker=_polza_enrichment_breaker,
    )


EnrichmentLlmClientDep = Annotated[LlmClient, Depends(get_enrichment_llm_client)]


def get_embedding_client(httpx_client: HttpClientDep) -> EmbeddingClient:
    settings = get_settings().yandex
    return EmbeddingClient(
        httpx_client=httpx_client,
        url=settings.yandex_embedding_api_url,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Api-Key {settings.yandex_api_key.get_secret_value()}',
        },
        circuit_breaker=_yandex_embedding_breaker,
        vector_dimension=settings.yandex_embedding_dim,
        rate_limiter=_yandex_embedding_rate_limiter,
    )


EmbeddingClientDep = Annotated[EmbeddingClient, Depends(get_embedding_client)]


def get_reranker_llm_client(httpx_client: HttpClientDep) -> LlmClient:
    """LLM-клиент для reranker'а (Этап 6) — Polza AI (Gemini), не Yandex.

    Тот же провайдер, что и у `get_enrichment_llm_client` (с 2026-06-21),
    но отдельный breaker и DI-инстанс — reranking в hot path поиска, не
    должен зависеть от состояния offline-обогащения при ingestion.
    """
    settings = get_settings().polza
    return LlmClient(
        httpx_client=httpx_client,
        model=settings.polza_reranker_llm_model,
        url=settings.polza_llm_api_url,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {settings.polza_api_key.get_secret_value()}',
        },
        timeout=settings.polza_reranker_timeout_seconds,
        retries=settings.polza_reranker_retries,
        circuit_breaker=_polza_reranker_breaker,
    )


RerankerLlmClientDep = Annotated[LlmClient, Depends(get_reranker_llm_client)]


def get_query_expansion_llm_client(httpx_client: HttpClientDep) -> LlmClient:
    """LLM-клиент расширения запроса (декомпозиция + переформулировка,
    раздел 8 плана) — Polza AI (Gemini), как и reranker/обогащение.

    В hot path поиска (вызывается до `hybrid_search`), поэтому отдельный
    breaker — отказ этого шага не должен влиять на состояние breaker'а
    reranker'а, и наоборот (LLM-2).
    """
    settings = get_settings().polza
    return LlmClient(
        httpx_client=httpx_client,
        model=settings.polza_query_expansion_llm_model,
        url=settings.polza_llm_api_url,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {settings.polza_api_key.get_secret_value()}',
        },
        timeout=settings.polza_query_expansion_timeout_seconds,
        retries=settings.polza_query_expansion_retries,
        circuit_breaker=_polza_query_expansion_breaker,
    )


QueryExpansionLlmClientDep = Annotated[LlmClient, Depends(get_query_expansion_llm_client)]
