from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient

from app.dependencies.auth import verify_api_key
from app.dependencies.services import get_ingestion_service
from app.exceptions.ingestion import EmptyIngestionContentError, StaleRevisionError
from app.main import app
from app.vectorstore.qdrant_client import QdrantVectorStore
from tests.unit.services.test_ingestion_service import _ingest_request, _make_service, _section_request


def _document(revision: str):
    return _ingest_request().model_copy(update={
        'revision_date': date.fromisoformat(revision),
        'raw_text': 'Статья 1. Первая\nПервоначальный текст.\n\nСтатья 2. Вторая\nДругой текст.',
    })


def _section(revision: str):
    return _section_request().model_copy(update={
        'revision_date': date.fromisoformat(revision),
        'raw_text': f'Статья 1. Текст редакции от {revision}.',
    })


@pytest.fixture
async def memory_service():
    service, *_ = _make_service()
    client = AsyncQdrantClient(':memory:')
    try:
        store = QdrantVectorStore(client, 'revision_ordering', 4)
        await store.ensure_collection()
        service.vector_store = store
        yield service
    finally:
        await client.close()


async def test_late_section_cannot_roll_back_new_revision(memory_service):
    service = memory_service
    initial = _document('2026-01-01')
    await service.ingest_document(initial)
    await service.ingest_section(initial.document_id, '1', _section('2026-03-01'))
    before = await service.vector_store.list_chunks(initial.document_id)
    llm_calls = service.llm_client.get_llm_response.await_count

    with pytest.raises(StaleRevisionError) as error:
        await service.ingest_section(initial.document_id, '1', _section('2026-02-01'))

    assert error.value.current_revision_date == date(2026, 3, 1)
    assert await service.vector_store.list_chunks(initial.document_id) == before
    assert service.llm_client.get_llm_response.await_count == llm_calls
    assert service.change_log_repository.add_entry.await_args.args[0].status == 'rejected'


async def test_repeated_section_delivery_keeps_same_chunks_and_history(memory_service):
    service = memory_service
    initial = _document('2026-01-01')
    await service.ingest_document(initial)
    first = await service.ingest_section(initial.document_id, '1', _section('2026-03-01'))
    before = await service.vector_store.list_chunks(initial.document_id)

    repeated = await service.ingest_section(initial.document_id, '1', _section('2026-03-01'))

    assert repeated.chunks_count == first.chunks_count
    assert repeated.superseded_chunks == 0
    assert await service.vector_store.list_chunks(initial.document_id) == before


async def test_new_revision_of_another_section_does_not_block_update(memory_service):
    service = memory_service
    initial = _document('2026-01-01')
    await service.ingest_document(initial)
    await service.ingest_section(initial.document_id, '1', _section('2026-03-01'))

    await service.ingest_section(initial.document_id, '2', _section('2026-02-01'))

    actual = [p for p in await service.vector_store.list_chunks(initial.document_id) if p['is_actual']]
    assert {(p['section_number'], p['effective_date']) for p in actual} == {
        ('1', '2026-03-01'), ('2', '2026-02-01'),
    }


async def test_full_reload_cannot_erase_newer_section(memory_service):
    service = memory_service
    initial = _document('2026-01-01')
    await service.ingest_document(initial)
    await service.ingest_section(initial.document_id, '1', _section('2026-03-01'))
    before = await service.vector_store.list_chunks(initial.document_id)

    with pytest.raises(StaleRevisionError):
        await service.ingest_document(_document('2026-02-01'))

    assert await service.vector_store.list_chunks(initial.document_id) == before


async def test_section_older_than_full_reload_is_rejected(memory_service):
    service = memory_service
    initial = _document('2026-03-01')
    await service.ingest_document(initial)
    before = await service.vector_store.list_chunks(initial.document_id)

    with pytest.raises(StaleRevisionError):
        await service.ingest_section(initial.document_id, '1', _section('2026-02-01'))

    assert await service.vector_store.list_chunks(initial.document_id) == before


async def test_retry_finishes_interrupted_revision_and_rejects_older_delivery(memory_service, monkeypatch):
    service = memory_service
    initial = _document('2026-01-01')
    await service.ingest_document(initial)
    set_inactive = service.vector_store.set_chunks_inactive

    async def fail_before_deactivation(**kwargs):
        raise RuntimeError('Обрыв после записи новых чанков')

    monkeypatch.setattr(service.vector_store, 'set_chunks_inactive', fail_before_deactivation)
    with pytest.raises(RuntimeError):
        await service.ingest_section(initial.document_id, '1', _section('2026-03-01'))

    with pytest.raises(StaleRevisionError):
        await service.ingest_section(initial.document_id, '1', _section('2026-02-01'))

    monkeypatch.setattr(service.vector_store, 'set_chunks_inactive', set_inactive)
    await service.ingest_section(initial.document_id, '1', _section('2026-03-01'))

    sections = [p for p in await service.vector_store.list_chunks(initial.document_id) if p['section_number'] == '1']
    assert {(p['version'], p['is_actual'], p['effective_until']) for p in sections} == {
        ('2026-01-01', False, '2026-03-01'), ('2026-03-01', True, None),
    }


@pytest.mark.parametrize('scope', ['section', 'document'])
async def test_empty_revision_cannot_remove_text_and_its_revision_date(memory_service, scope):
    service = memory_service
    initial = _document('2026-03-01')
    await service.ingest_document(initial)
    before = await service.vector_store.list_chunks(initial.document_id)
    note = '(Статья 1 в редакции Федерального закона от 01.04.2026 № 1-ФЗ)'

    with pytest.raises(EmptyIngestionContentError):
        if scope == 'section':
            await service.ingest_section(
                initial.document_id, '1', _section('2026-04-01').model_copy(update={'raw_text': note}),
            )
        else:
            await service.ingest_document(_document('2026-04-01').model_copy(update={'raw_text': note}))

    assert await service.vector_store.list_chunks(initial.document_id) == before


@pytest.mark.parametrize('scope', ['section', 'document'])
async def test_api_returns_structured_stale_revision_conflict(memory_service, scope):
    service = memory_service
    initial = _document('2026-03-01')
    await service.ingest_document(initial)
    previous_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[get_ingestion_service] = lambda: service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            if scope == 'section':
                response = await client.put(
                    f'/api/v1/document/{initial.document_id}/sections/1',
                    json=_section('2026-02-01').model_dump(mode='json'),
                )
            else:
                response = await client.post('/api/v1/ingest', json=_document('2026-02-01').model_dump(mode='json'))
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)

    assert response.status_code == 409
    assert response.json()['detail']['code'] == 'stale_revision'
    assert response.json()['detail']['current_revision_date'] == '2026-03-01'
