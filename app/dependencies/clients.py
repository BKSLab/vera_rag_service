from typing import Annotated

from fastapi import Depends

from app.clients.embeddings import EmbeddingClient
from app.clients.llm import LlmClient
from app.core.circuit_breaker import CircuitBreaker
from app.core.settings import get_settings
from app.dependencies.http_client import HttpClientDep

# LLM-2 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — один breaker на
# провайдера+use-case, не на запрос: должен переживать конкретный
# `LlmClient`/`EmbeddingClient`, который DI создаёт заново на каждый
# HTTP-запрос. Три независимых breaker'а — отказ Yandex enrichment не
# должен "размыкать" вызовы к Polza reranker'у и наоборот.
_yandex_llm_breaker = CircuitBreaker()
_yandex_embedding_breaker = CircuitBreaker()
_polza_reranker_breaker = CircuitBreaker()


def get_llm_client(httpx_client: HttpClientDep) -> LlmClient:
    settings = get_settings().yandex
    return LlmClient(
        httpx_client=httpx_client,
        model=settings.llm_model_uri,
        url=settings.yandex_llm_api_url,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Api-Key {settings.yandex_api_key.get_secret_value()}',
        },
        # LLM-1 — обходные пути под капризы конкретно YandexGPT (markdown
        # code fence/эмфазис в JSON-ответе), не включаются для других
        # провайдеров (см. get_reranker_llm_client).
        strip_markdown_artifacts=True,
        circuit_breaker=_yandex_llm_breaker,
    )


LlmClientDep = Annotated[LlmClient, Depends(get_llm_client)]


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
    )


EmbeddingClientDep = Annotated[EmbeddingClient, Depends(get_embedding_client)]


def get_reranker_llm_client(httpx_client: HttpClientDep) -> LlmClient:
    """LLM-клиент для reranker'а (Этап 6) — Polza AI (Gemini), не Yandex.

    Намеренно отдельная модель/провайдер от enrichment (Этап 3): reranking —
    hot path поиска, важна стабильность JSON-вывода без специфичных под
    YandexGPT доработок промпта; Gemini через Polza соблюдает JSON-формат
    из коробки (см. RAG_SERVICE_PLAN.md, Этап 6).
    """
    settings = get_settings().polza
    return LlmClient(
        httpx_client=httpx_client,
        model=settings.polza_llm_model,
        url=settings.polza_llm_api_url,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {settings.polza_api_key.get_secret_value()}',
        },
        circuit_breaker=_polza_reranker_breaker,
    )


RerankerLlmClientDep = Annotated[LlmClient, Depends(get_reranker_llm_client)]
