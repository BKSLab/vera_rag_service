import json

import httpx

from app.core.settings import PolzaSettings, get_settings
from app.dependencies.clients import (
    get_enrichment_llm_client,
    get_query_expansion_llm_client,
    get_reranker_llm_client,
)
from app.models.schemas import ChunkEnrichmentResult


async def test_enrichment_sends_strict_json_schema_to_provider():
    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
            'synthetic_title': 'Проверка формата',
            'hypothetical_questions': ['Кто принимает решение?', 'Куда обратиться?', 'Какие нужны документы?'],
        })}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as httpx_client:
        client = get_enrichment_llm_client(httpx_client)
        result = await client.get_llm_response(
            content='Фрагмент документа', prompt='Верни JSON.', schema=ChunkEnrichmentResult,
        )

        assert 'response_format' not in get_query_expansion_llm_client(httpx_client).extra_payload
        assert 'response_format' not in get_reranker_llm_client(httpx_client).extra_payload

    assert isinstance(result, ChunkEnrichmentResult)
    response_format = captured_payload['response_format']
    assert response_format['type'] == 'json_schema'
    assert response_format['json_schema']['strict'] is True
    schema = response_format['json_schema']['schema']
    assert schema['additionalProperties'] is False
    assert set(schema['required']) == {'synthetic_title', 'hypothetical_questions'}
    questions = schema['properties']['hypothetical_questions']
    assert questions['items']['type'] == 'string'
    assert questions['minItems'] == 3
    assert questions['maxItems'] == 5


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
