from types import SimpleNamespace
from unittest.mock import AsyncMock

from app import main


class _ConnectionContext:
    async def __aenter__(self):
        return SimpleNamespace(execute=AsyncMock())

    async def __aexit__(self, exc_type, exc, traceback):
        return None


async def test_lifespan_configures_and_shuts_down_tracing(monkeypatch):
    engine = SimpleNamespace(connect=lambda: _ConnectionContext(), dispose=AsyncMock())
    vector_store = SimpleNamespace(ensure_collection=AsyncMock())
    qdrant = SimpleNamespace(close=AsyncMock())
    http_client = SimpleNamespace(aclose=AsyncMock())
    calls = []

    monkeypatch.setattr(main, 'engine', engine)
    monkeypatch.setattr(main, 'get_vector_store', lambda: vector_store)
    monkeypatch.setattr(main, 'qdrant_client', qdrant)
    monkeypatch.setattr(main, 'external_api_http_client', http_client)
    monkeypatch.setattr(main, 'configure_tracing', lambda settings: calls.append('configure'))
    monkeypatch.setattr(main, 'shutdown_tracing', lambda: calls.append('shutdown'))

    async with main.lifespan(main.app):
        assert calls == ['configure']

    assert calls == ['configure', 'shutdown']
    engine.dispose.assert_awaited_once()
    qdrant.close.assert_awaited_once()
    http_client.aclose.assert_awaited_once()
