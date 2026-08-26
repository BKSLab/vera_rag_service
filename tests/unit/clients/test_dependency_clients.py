import httpx

from app.core.settings import PolzaSettings, get_settings
from app.dependencies.clients import (
    get_enrichment_llm_client,
    get_query_expansion_llm_client,
    get_reranker_llm_client,
)


async def test_polza_llm_clients_use_separate_temperature_timeout_and_retry_settings():
    settings = get_settings().polza
    async with httpx.AsyncClient() as httpx_client:
        enrichment_client = get_enrichment_llm_client(httpx_client)
        query_expansion_client = get_query_expansion_llm_client(httpx_client)
        reranker_client = get_reranker_llm_client(httpx_client)

    assert enrichment_client.timeout == settings.polza_enrichment_timeout_seconds
    assert enrichment_client.retries == settings.polza_enrichment_retries
    assert enrichment_client.temperature == settings.polza_enrichment_llm_temperature == 0.3
    assert query_expansion_client.timeout == settings.polza_query_expansion_timeout_seconds
    assert query_expansion_client.retries == settings.polza_query_expansion_retries
    assert query_expansion_client.temperature == settings.polza_query_expansion_llm_temperature == 0.0
    assert reranker_client.timeout == settings.polza_reranker_timeout_seconds
    assert reranker_client.retries == settings.polza_reranker_retries
    assert reranker_client.temperature == settings.polza_reranker_llm_temperature == 0.0
    assert query_expansion_client.timeout < enrichment_client.timeout
    assert reranker_client.timeout < enrichment_client.timeout


def test_polza_llm_defaults_use_stable_gemini_flash_lite_model():
    assert PolzaSettings.model_fields['polza_enrichment_llm_model'].default == 'google/gemini-3.5-flash-lite'
    assert PolzaSettings.model_fields['polza_reranker_llm_model'].default == 'google/gemini-3.5-flash-lite'
    assert PolzaSettings.model_fields['polza_query_expansion_llm_model'].default == 'google/gemini-3.5-flash-lite'


def test_polza_llm_temperature_defaults_are_component_specific():
    assert PolzaSettings.model_fields['polza_enrichment_llm_temperature'].default == 0.3
    assert PolzaSettings.model_fields['polza_reranker_llm_temperature'].default == 0.0
    assert PolzaSettings.model_fields['polza_query_expansion_llm_temperature'].default == 0.0
