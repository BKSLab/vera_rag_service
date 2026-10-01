import asyncio
from datetime import date

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.exceptions.ingestion import StaleRevisionError
from app.vectorstore.qdrant_client import QdrantVectorStore
from tests.integration.services.test_ingestion_service import make_ingestion_service
from tests.unit.services.test_revision_ordering import _document, _section


@pytest.fixture
async def memory_store():
    client = AsyncQdrantClient(':memory:')
    try:
        store = QdrantVectorStore(client, 'revision_concurrency', 4)
        await store.ensure_collection()
        yield store
    finally:
        await client.close()


@pytest.mark.parametrize(('first_kind', 'second_kind'), [
    ('section', 'section'), ('document', 'section'), ('section', 'document'),
])
async def test_revision_check_waits_for_other_writer_even_after_its_commit(
    db_session, memory_store, monkeypatch, first_kind, second_kind,
):
    factory = async_sessionmaker(db_session.bind, expire_on_commit=False)
    initial = _document('2026-01-01')
    committed = asyncio.Event()
    finish_first = asyncio.Event()
    async with factory() as first_session, factory() as second_session:
        first = make_ingestion_service(memory_store, first_session)
        second = make_ingestion_service(memory_store, second_session)
        await first.ingest_document(initial)

        method_name = '_save_document_record' if first_kind == 'document' else '_write_change_log_entry'
        write_record = getattr(first, method_name)

        async def pause_after_commit(**kwargs):
            await write_record(**kwargs)
            committed.set()
            await finish_first.wait()

        monkeypatch.setattr(first, method_name, pause_after_commit)

        async def ingest(service, kind, revision):
            if kind == 'document':
                return await service.ingest_document(_document(revision))
            return await service.ingest_section(initial.document_id, '1', _section(revision))

        first_task = asyncio.create_task(ingest(first, first_kind, '2026-03-01'))
        second_task = None
        try:
            await asyncio.wait_for(committed.wait(), timeout=10)
            second_task = asyncio.create_task(ingest(second, second_kind, '2026-02-01'))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(second_task), timeout=0.2)
            finish_first.set()
            await asyncio.wait_for(first_task, timeout=10)
            with pytest.raises(StaleRevisionError):
                await asyncio.wait_for(second_task, timeout=10)
        finally:
            finish_first.set()
            tasks = [first_task] + ([second_task] if second_task is not None else [])
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # Выход по отказу также должен освобождать блокировку.
        await asyncio.wait_for(ingest(second, 'section', '2026-04-01'), timeout=10)
        assert await memory_store.get_latest_revision_date(initial.document_id) == date(2026, 4, 1)


async def test_cancelled_writer_releases_document_lock(db_session, memory_store):
    factory = async_sessionmaker(db_session.bind, expire_on_commit=False)
    entered = asyncio.Event()
    initial = _document('2026-01-01')
    async with factory() as first_session, factory() as second_session:
        first = make_ingestion_service(memory_store, first_session)
        second = make_ingestion_service(memory_store, second_session)

        async def hold_lock():
            async with first.document_repository.document_lock(initial.document_id):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold_lock())
        try:
            await asyncio.wait_for(entered.wait(), timeout=10)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        await asyncio.wait_for(second.ingest_document(initial), timeout=10)
