import asyncio

import pytest
from opentelemetry.trace import StatusCode

from app.core.request_context import set_request_id
from app.core.settings import ObservabilitySettings
from app.exceptions.embedding import EmbeddingApiRequestError
from app.exceptions.llm import LlmApiRequestError
from app.exceptions.search_log import SearchLogRepositoryError
from app.models.schemas import RerankResult, SearchFilters
from app.observability.tracing import (
    _add_exporter,
    _create_otlp_exporter,
    shutdown_tracing,
)
from tests.unit.services.test_search_service import _build_service


def _span(exporter):
    return next(span for span in exporter.get_finished_spans() if span.name == 'rag.search')


async def test_success_span_has_aggregates_and_no_sensitive_content(telemetry_exporter):
    set_request_id('request-success')
    service, repository = _build_service()

    results = await service.search(
        query='квота на инвалидов',
        filters=SearchFilters(audience='employer'),
        top_k=5,
    )

    span = _span(telemetry_exporter)
    saved_log = repository.save_search_log.await_args.args[0]
    assert len(results) == 1
    assert span.attributes['request.id'] == saved_log.request_id == 'request-success'
    assert span.attributes['openinference.span.kind'] == 'RETRIEVER'
    assert span.attributes['rag.query_variant_count'] == 1
    assert span.attributes['rag.dense_candidate_count'] >= 1
    assert span.attributes['rag.sparse_candidate_count'] >= 1
    assert span.attributes['rag.rrf_candidate_count'] == 1
    assert span.attributes['rag.result_chunk_count'] == 1
    assert span.attributes['rag.query_expansion.status'] == 'ok'
    assert span.attributes['rag.reranker.status'] == 'ok'
    assert span.attributes['rag.search_log.status'] == 'ok'
    assert span.attributes['rag.outcome'] == 'ok'
    serialized_attributes = str(span.attributes)
    assert 'квота на инвалидов' not in serialized_attributes
    assert 'Квота на трудоустройство инвалидов' not in serialized_attributes


async def test_no_candidates_is_empty_not_error(telemetry_exporter):
    service, _ = _build_service(has_candidates=False)

    assert await service.search('no results', SearchFilters(), 5) == []

    span = _span(telemetry_exporter)
    assert span.attributes['rag.outcome'] == 'empty'
    assert span.attributes['rag.reranker.status'] == 'no_candidates'
    assert span.status.status_code is StatusCode.UNSET


@pytest.mark.parametrize('fallback_stage', ['query_expansion', 'reranker'])
async def test_expected_fallback_is_degraded_not_error(telemetry_exporter, fallback_stage):
    service, _ = _build_service()
    error = LlmApiRequestError('temporary', 'https://llm.test')
    if fallback_stage == 'query_expansion':
        service.query_expansion_llm_client.get_llm_response.side_effect = error
    else:
        service.reranker_llm_client.get_llm_response.side_effect = error

    results = await service.search('fallback query', SearchFilters(), 5)

    span = _span(telemetry_exporter)
    assert results
    assert span.attributes['rag.outcome'] == 'degraded'
    assert span.status.status_code is StatusCode.UNSET


async def test_no_relevant_result_is_empty_not_error(telemetry_exporter):
    service, _ = _build_service()
    service.reranker_llm_client.get_llm_response.return_value = RerankResult(ranked_indices=[])

    assert await service.search('unrelated', SearchFilters(), 5) == []

    span = _span(telemetry_exporter)
    assert span.attributes['rag.outcome'] == 'empty'
    assert span.attributes['rag.reranker.status'] == 'no_relevant'
    assert span.status.status_code is StatusCode.UNSET


async def test_embedding_failure_records_terminal_error(telemetry_exporter):
    service, _ = _build_service()
    service.embedding_client.get_embedding.side_effect = EmbeddingApiRequestError(
        'timeout', 'https://embedding.test'
    )

    with pytest.raises(EmbeddingApiRequestError):
        await service.search('query', SearchFilters(), 5)

    span = _span(telemetry_exporter)
    assert span.attributes['rag.outcome'] == 'error'
    assert span.status.status_code is StatusCode.ERROR


async def test_search_log_failure_does_not_fail_search_span(telemetry_exporter):
    service, repository = _build_service()
    repository.save_search_log.side_effect = SearchLogRepositoryError('database unavailable')

    results = await service.search('query', SearchFilters(), 5)

    span = _span(telemetry_exporter)
    assert results
    assert span.attributes['rag.search_log.status'] == 'unavailable'
    assert span.attributes['rag.outcome'] == 'ok'
    assert span.status.status_code is StatusCode.UNSET


async def test_parallel_searches_have_isolated_trace_and_request_ids(telemetry_exporter):
    async def run(request_id: str):
        set_request_id(request_id)
        service, repository = _build_service()
        await service.search('same query', SearchFilters(), 5)
        return repository.save_search_log.await_args.args[0].request_id

    saved_ids = await asyncio.gather(run('request-a'), run('request-b'))

    spans = [span for span in telemetry_exporter.get_finished_spans() if span.name == 'rag.search']
    assert set(saved_ids) == {'request-a', 'request-b'}
    assert {span.attributes['request.id'] for span in spans} == {'request-a', 'request-b'}
    assert len({span.context.trace_id for span in spans}) == 2


def test_project_header_disabled_mode_and_shutdown(monkeypatch):
    exporter_calls = {}

    class _Exporter:
        def __init__(self, **kwargs):
            exporter_calls.update(kwargs)

    monkeypatch.setattr('app.observability.tracing.OTLPSpanExporter', _Exporter)
    _create_otlp_exporter(ObservabilitySettings(phoenix_project_name='vera-testing'))
    assert exporter_calls['headers'] == {'x-project-name': 'vera-testing'}

    class _NoExporterProvider:
        def add_span_processor(self, processor):
            raise AssertionError('disabled Phoenix must not add processor')

    _add_exporter(_NoExporterProvider(), ObservabilitySettings(phoenix_enabled=False))

    calls = []

    class _Provider:
        def force_flush(self, timeout_millis):
            calls.append(('flush', timeout_millis))
            return True

        def shutdown(self):
            calls.append(('shutdown', None))

    monkeypatch.setattr('app.observability.tracing._provider', _Provider())
    monkeypatch.setattr('app.observability.tracing._shutdown', False)
    shutdown_tracing()
    shutdown_tracing()
    assert calls == [('flush', 10_000), ('shutdown', None)]
