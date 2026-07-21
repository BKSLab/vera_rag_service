from unittest.mock import AsyncMock

from httpx import AsyncClient
from opentelemetry import propagate, trace

from app.dependencies.services import get_health_service, get_search_service
from app.main import app
from app.observability.tracing import get_tracer
from app.schemas.health import HealthSchema
from app.services.health import HealthService
from tests.unit.services.test_search_service import _build_service


async def test_http_traceparent_becomes_rag_search_remote_parent(
    async_client: AsyncClient, telemetry_exporter
):
    service, repository = _build_service()
    app.dependency_overrides[get_search_service] = lambda: service
    carrier = {}
    with get_tracer().start_as_current_span('mcp.execute.vera_rag_kb') as parent:
        parent_context = parent.get_span_context()
        propagate.inject(carrier)

    response = await async_client.post(
        '/api/v1/search',
        json={'query': 'safe test query', 'top_k': 5},
        headers={**carrier, 'X-Request-ID': 'mcp-request-id'},
    )

    assert response.status_code == 200
    rag_span = next(
        span for span in telemetry_exporter.get_finished_spans() if span.name == 'rag.search'
    )
    saved_log = repository.save_search_log.await_args.args[0]
    assert rag_span.context.trace_id == parent_context.trace_id
    assert rag_span.parent.span_id == parent_context.span_id
    assert rag_span.parent.is_remote is True
    assert rag_span.attributes['request.id'] == saved_log.request_id == 'mcp-request-id'
    assert trace.get_current_span().get_span_context().is_valid is False


async def test_health_and_metrics_do_not_create_application_spans(
    async_client: AsyncClient, telemetry_exporter
):
    health_service = AsyncMock(spec=HealthService)
    health_service.check_health.return_value = HealthSchema(status='ok', database='ok')
    app.dependency_overrides[get_health_service] = lambda: health_service

    health_response = await async_client.get('/api/v1/health')
    metrics_response = await async_client.get('/metrics')

    assert health_response.status_code == 200
    assert metrics_response.status_code == 200
    assert telemetry_exporter.get_finished_spans() == ()
