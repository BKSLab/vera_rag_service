import httpx

from app.core.settings import get_settings
from app.dependencies.clients import (
    get_enrichment_llm_client,
    get_query_expansion_llm_client,
    get_reranker_llm_client,
)


async def test_polza_llm_clients_use_separate_timeout_and_retry_settings():
    settings = get_settings().polza
    async with httpx.AsyncClient() as httpx_client:
        enrichment_client = get_enrichment_llm_client(httpx_client)
        query_expansion_client = get_query_expansion_llm_client(httpx_client)
        reranker_client = get_reranker_llm_client(httpx_client)

    assert enrichment_client.timeout == settings.polza_enrichment_timeout_seconds
    assert enrichment_client.retries == settings.polza_enrichment_retries
    assert query_expansion_client.timeout == settings.polza_query_expansion_timeout_seconds
    assert query_expansion_client.retries == settings.polza_query_expansion_retries
    assert reranker_client.timeout == settings.polza_reranker_timeout_seconds
    assert reranker_client.retries == settings.polza_reranker_retries
    assert query_expansion_client.timeout < enrichment_client.timeout
    assert reranker_client.timeout < enrichment_client.timeout
