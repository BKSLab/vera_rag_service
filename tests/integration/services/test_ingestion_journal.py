import asyncio
import hashlib
import json
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.request_context import set_request_id
from app.db.models.ingestion_run import IngestionRun
from app.exceptions.document import DocumentRepositoryError
from app.exceptions.ingestion import IngestionIntegrityError, StaleRevisionError
from app.repositories.ingestion_run import IngestionRunRepository
from app.vectorstore.qdrant_client import QdrantVectorStore
from tests.integration.services.test_ingestion_service import make_ingestion_service
from tests.unit.services.test_revision_ordering import _document, _section


@pytest.fixture
async def journal_service(db_session):
    client = AsyncQdrantClient(':memory:')
    store = QdrantVectorStore(client, 'trace_test_collection', 4)
    await store.ensure_collection()
    service = make_ingestion_service(store, db_session)
    factory = async_sessionmaker(db_session.bind, expire_on_commit=False)
    service.ingestion_run_repository = IngestionRunRepository(factory)
    try:
        yield service, factory
    finally:
        await client.close()


async def runs(factory):
    async with factory() as session:
        return (await session.scalars(select(IngestionRun).order_by(IngestionRun.started_at))).all()


async def test_document_and_section_have_verified_receipts_and_durable_history(journal_service, db_session):
    service, factory = journal_service
    document = _document('2026-01-01')
    loaded = await service.ingest_document(document)
    service.source, service.change_id = 'legal_sync', 42
    set_request_id('delivery-attempt-42')
    try:
        request = _section('2026-02-01')
        updated = await service.ingest_section(document.document_id, '1', request)
    finally:
        set_request_id(None)
    await db_session.rollback()
    first, second = await runs(factory)
    assert first.id == loaded.operation_id and second.id == updated.operation_id
    assert first.id != second.id
    assert first.status == second.status == 'succeeded'
    assert second.request_id == 'delivery-attempt-42' and second.change_id == 42
    assert second.source == 'legal_sync'
    assert second.collection_name == updated.collection_name == 'trace_test_collection'
    assert updated.integrity_verified is True
    assert updated.input_sha256 == hashlib.sha256(request.raw_text.encode()).hexdigest()
    stages = {step['key']: step for step in second.stages}
    assert stages['integrity']['details']['identifiers_match'] is True
    assert stages['retire']['details']['superseded_chunks'] >= 1
    assert stages['embedding']['details']['completed'] == updated.chunks_count
    assert request.raw_text not in json.dumps(second.result)
    assert all(step.get('duration_seconds', 0) >= 0 for step in second.stages)


async def test_live_stage_is_visible_and_cancellation_does_not_erase_it(journal_service):
    service, factory = journal_service
    entered = asyncio.Event()

    async def hold_embedding(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    service.embedding_client.get_embedding.side_effect = hold_embedding
    task = asyncio.create_task(service.ingest_document(_document('2026-01-01')))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        row, = await runs(factory)
        assert row.status == 'running' and row.current_stage == 'embedding'
        assert next(step for step in row.stages if step['key'] == 'embedding')['details']['total'] > 0
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    row, = await runs(factory)
    assert row.status == 'interrupted' and row.finished_at is not None
    assert next(step for step in row.stages if step['key'] == 'upsert')['status'] == 'not_run'


async def test_integrity_failure_records_cleanup_and_no_success(journal_service, monkeypatch):
    service, factory = journal_service
    monkeypatch.setattr(service.vector_store, 'get_actual_document_chunk_ids', AsyncMock(return_value=[]))
    with pytest.raises(IngestionIntegrityError):
        await service.ingest_document(_document('2026-01-01'))
    row, = await runs(factory)
    stages = {step['key']: step for step in row.stages}
    assert row.status == 'failed'
    assert stages['integrity']['status'] == 'failed'
    assert stages['cleanup']['status'] == 'succeeded'
    assert stages['registry']['status'] == 'not_run'
    assert await service.vector_store.get_document_versions(row.document_id) == []


async def test_stale_update_has_rejection_before_paid_calls(journal_service):
    service, factory = journal_service
    request = _document('2026-03-01')
    await service.ingest_document(request)
    service.llm_client.get_llm_response.reset_mock()
    with pytest.raises(StaleRevisionError):
        await service.ingest_section(request.document_id, '1', _section('2026-01-01'))
    row = (await runs(factory))[-1]
    assert row.status == 'rejected'
    assert row.current_stage == 'revision'
    assert next(step for step in row.stages if step['key'] == 'change_log')['status'] == 'succeeded'
    assert next(step for step in row.stages if step['key'] == 'revision')['status'] == 'failed'
    service.llm_client.get_llm_response.assert_not_awaited()


async def test_registry_failure_is_warning_with_successful_collection(journal_service, monkeypatch):
    service, factory = journal_service
    monkeypatch.setattr(service.document_repository, 'save_document', AsyncMock(side_effect=DocumentRepositoryError('registry unavailable')))
    result = await service.ingest_document(_document('2026-01-01'))
    row, = await runs(factory)
    assert result.status == row.status == 'warning'
    assert result.integrity_verified is True
    assert any('registry unavailable' in warning for warning in result.warnings)
    assert next(step for step in row.stages if step['key'] == 'registry')['status'] == 'warning'


async def test_journal_failure_does_not_mask_successful_write(journal_service):
    service, _ = journal_service
    service.ingestion_run_repository = AsyncMock(spec=IngestionRunRepository)
    service.ingestion_run_repository.save.side_effect = OSError('journal unavailable')
    result = await service.ingest_document(_document('2026-01-01'))
    assert result.status == 'warning' and result.integrity_verified
    assert result.warnings
