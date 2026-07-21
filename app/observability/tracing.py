import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from app.core.settings import ObservabilitySettings

SERVICE_NAME = 'vera_rag_service'
logger = logging.getLogger(SERVICE_NAME)

_provider: TracerProvider | None = None
_shutdown = False


def configure_tracing(settings: ObservabilitySettings) -> TracerProvider:
    global _provider, _shutdown
    if _provider is not None:
        return _provider

    provider = TracerProvider(resource=Resource.create({'service.name': SERVICE_NAME}))
    _add_exporter(provider, settings)
    trace.set_tracer_provider(provider)
    _provider = provider
    _shutdown = False
    return provider


def _create_otlp_exporter(settings: ObservabilitySettings) -> OTLPSpanExporter:
    return OTLPSpanExporter(
        endpoint=settings.phoenix_otlp_endpoint,
        headers={'x-project-name': settings.phoenix_project_name},
    )


def _add_exporter(provider: TracerProvider, settings: ObservabilitySettings) -> None:
    if settings.phoenix_enabled:
        provider.add_span_processor(BatchSpanProcessor(_create_otlp_exporter(settings)))


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(SERVICE_NAME)


def force_flush_tracing(timeout_millis: int = 10_000) -> bool:
    if _provider is None or _shutdown:
        return True
    try:
        return _provider.force_flush(timeout_millis=timeout_millis)
    except Exception:  # noqa: BLE001 - telemetry не должна ломать shutdown
        logger.exception('Не удалось выполнить force_flush OpenTelemetry')
        return False


def shutdown_tracing(timeout_millis: int = 10_000) -> None:
    global _shutdown
    if _provider is None or _shutdown:
        return
    force_flush_tracing(timeout_millis=timeout_millis)
    try:
        _provider.shutdown()
    except Exception:  # noqa: BLE001 - telemetry не должна ломать shutdown
        logger.exception('Не удалось завершить OpenTelemetry provider')
    finally:
        _shutdown = True


def reset_for_tests(exporter: SpanExporter | None = None) -> TracerProvider:
    global _provider, _shutdown
    _provider = None
    provider = TracerProvider(resource=Resource.create({'service.name': SERVICE_NAME}))
    if exporter is not None:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _provider = provider
    _shutdown = False
    return provider
