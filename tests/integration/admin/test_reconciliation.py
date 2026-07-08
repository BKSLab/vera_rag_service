from datetime import date
from uuid import uuid4

import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from app.admin.reconciliation import find_active_documents_missing_in_qdrant
from app.core.settings import get_settings
from app.db.models.document import Document
from app.models.schemas import Chunk, DocumentMetadataInput, EmbeddedChunk, EnrichedChunk
from app.vectorstore.qdrant_client import QdrantVectorStore

VECTOR_DIM = 4


@pytest_asyncio.fixture
async def vector_store():
    settings = get_settings().qdrant
    client = AsyncQdrantClient(url=settings.qdrant_url)
    collection_name = f'test_{uuid4().hex}'
    store = QdrantVectorStore(client=client, collection_name=collection_name, vector_dim=VECTOR_DIM)
    await store.ensure_collection()

    yield store

    await client.delete_collection(collection_name)
    await client.close()


def make_document_row(document_id: str, version: str) -> Document:
    return Document(
        document_id=document_id, version=version, category='labor_code',
        source_title='Источник', audience='both', topics=['quota'],
        effective_date=date(2026, 1, 1), is_active=True,
    )


async def test_find_active_documents_missing_in_qdrant_returns_empty_when_consistent(vector_store, db_session):
    document_id = f'doc-{uuid4().hex}'
    db_session.add(make_document_row(document_id, '2026-01-01'))
    await db_session.commit()

    chunk = Chunk(
        chunk_id=str(uuid4()), chunk_index=0, chunk_number_in_section=0,
        document_id=document_id, parent_id=f'{document_id}:21', category='labor_code',
        section_index=0, section_number='21', section_title='Статья 21', text='Текст.',
    )
    enriched = EnrichedChunk(chunk=chunk, synthetic_title='Заголовок', hypothetical_questions=['В1?', 'В2?', 'В3?'])
    embedded = EmbeddedChunk(enriched_chunk=enriched, chunk_vector=[0.1, 0.2, 0.3, 0.4], question_vectors=[])
    metadata = DocumentMetadataInput(
        source_title='Источник', audience='both', topics=['quota'], version='2026-01-01', effective_date=date(2026, 1, 1)
    )
    await vector_store.upsert_chunk(embedded, metadata)

    mismatches = await find_active_documents_missing_in_qdrant(db_session, vector_store)

    assert mismatches == []


async def test_find_active_documents_missing_in_qdrant_detects_orphaned_registry_row(vector_store, db_session):
    """ING-5 — реестр говорит "активна", но в Qdrant для этой версии нет
    ни одного чанка (например, отказ на середине upsert_chunks)."""
    document_id = f'doc-{uuid4().hex}'
    db_session.add(make_document_row(document_id, '2026-01-01'))
    await db_session.commit()

    mismatches = await find_active_documents_missing_in_qdrant(db_session, vector_store)

    assert mismatches == [(document_id, '2026-01-01')]


async def test_find_active_documents_missing_in_qdrant_detects_version_without_actual_chunks(vector_store, db_session):
    """Версия может существовать в Qdrant только историческими чанками
    (`is_actual=False`). Для поиска это всё равно отсутствие документа."""
    document_id = f'doc-{uuid4().hex}'
    db_session.add(make_document_row(document_id, '2026-01-01'))
    await db_session.commit()

    chunk = Chunk(
        chunk_id=str(uuid4()), chunk_index=0, chunk_number_in_section=0,
        document_id=document_id, parent_id=f'{document_id}:21', category='labor_code',
        section_index=0, section_number='21', section_title='Статья 21', text='Текст.',
    )
    enriched = EnrichedChunk(chunk=chunk, synthetic_title='Заголовок', hypothetical_questions=['В1?', 'В2?', 'В3?'])
    embedded = EmbeddedChunk(enriched_chunk=enriched, chunk_vector=[0.1, 0.2, 0.3, 0.4], question_vectors=[])
    metadata = DocumentMetadataInput(
        source_title='Источник', audience='both', topics=['quota'], version='2026-01-01', effective_date=date(2026, 1, 1)
    )
    await vector_store.upsert_chunk(embedded, metadata)
    await vector_store.set_chunks_inactive([chunk.chunk_id], effective_until=date(2026, 2, 1))

    mismatches = await find_active_documents_missing_in_qdrant(db_session, vector_store)

    assert mismatches == [(document_id, '2026-01-01')]
