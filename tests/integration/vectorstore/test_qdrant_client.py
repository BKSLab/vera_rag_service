from datetime import date
from uuid import uuid4

import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from app.core.settings import get_settings
from app.models.schemas import Chunk, DocumentMetadataInput, EmbeddedChunk, EnrichedChunk
from app.vectorstore.qdrant_client import (
    CHUNK_VECTOR_NAME,
    QUESTION_VECTOR_NAMES,
    QdrantVectorStore,
)

VECTOR_DIM = 4


@pytest_asyncio.fixture
async def vector_store():
    """Тестовая коллекция на реальном локальном Qdrant (docker-compose), с очисткой после теста."""
    settings = get_settings().qdrant
    client = AsyncQdrantClient(url=settings.qdrant_url)
    collection_name = f'test_{uuid4().hex}'
    store = QdrantVectorStore(client=client, collection_name=collection_name, vector_dim=VECTOR_DIM)

    yield store

    await client.delete_collection(collection_name)
    await client.close()


def make_embedded_chunk(document_id: str = 'fz-181', chunk_index: int = 0, questions: int = 3) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=str(uuid4()),
        chunk_index=chunk_index,
        document_id=document_id,
        category='labor_code',
        section_index=0,
        section_number='21',
        section_title='Статья 21',
        text='Текст чанка.',
    )
    enriched_chunk = EnrichedChunk(
        chunk=chunk,
        synthetic_title='Заголовок',
        hypothetical_questions=[f'Вопрос {i}?' for i in range(questions)],
    )
    return EmbeddedChunk(
        enriched_chunk=enriched_chunk,
        chunk_vector=[0.1, 0.2, 0.3, 0.4],
        question_vectors=[[0.1 * (i + 1)] * VECTOR_DIM for i in range(questions)],
    )


def make_document_metadata() -> DocumentMetadataInput:
    return DocumentMetadataInput(
        source_title='ФЗ-181, Статья 21',
        audience='both',
        topic='quota',
        version='2026-01-01',
        effective_date=date(2026, 1, 1),
    )


async def test_ensure_collection_creates_collection_with_named_vectors(vector_store):
    await vector_store.ensure_collection()

    info = await vector_store.client.get_collection(vector_store.collection_name)
    vector_names = set(info.config.params.vectors.keys())

    assert vector_names == {CHUNK_VECTOR_NAME, *QUESTION_VECTOR_NAMES}


async def test_ensure_collection_is_idempotent(vector_store):
    await vector_store.ensure_collection()
    await vector_store.ensure_collection()

    info = await vector_store.client.get_collection(vector_store.collection_name)
    assert info is not None


async def test_upsert_chunk_makes_point_retrievable_with_metadata_payload(vector_store):
    await vector_store.ensure_collection()
    embedded_chunk = make_embedded_chunk(questions=3)
    document_metadata = make_document_metadata()

    await vector_store.upsert_chunk(embedded_chunk, document_metadata)

    chunk_id = embedded_chunk.enriched_chunk.chunk.chunk_id
    points = await vector_store.client.retrieve(
        collection_name=vector_store.collection_name, ids=[chunk_id], with_payload=True, with_vectors=True
    )

    assert len(points) == 1
    point = points[0]
    assert point.payload['document_id'] == 'fz-181'
    assert point.payload['audience'] == 'both'
    assert point.payload['topic'] == 'quota'
    assert CHUNK_VECTOR_NAME in point.vector
    assert QUESTION_VECTOR_NAMES[0] in point.vector
    assert QUESTION_VECTOR_NAMES[2] in point.vector
    assert QUESTION_VECTOR_NAMES[3] not in point.vector


async def test_delete_document_removes_all_its_chunks(vector_store):
    await vector_store.ensure_collection()
    document_metadata = make_document_metadata()
    chunks = [make_embedded_chunk(document_id='fz-181', chunk_index=i) for i in range(3)]
    other_document_chunk = make_embedded_chunk(document_id='fz-999', chunk_index=0)

    await vector_store.upsert_chunks(chunks, document_metadata)
    await vector_store.upsert_chunk(other_document_chunk, document_metadata)

    await vector_store.delete_document('fz-181')

    remaining_ids = [c.enriched_chunk.chunk.chunk_id for c in chunks]
    remaining = await vector_store.client.retrieve(
        collection_name=vector_store.collection_name, ids=remaining_ids
    )
    other_remaining = await vector_store.client.retrieve(
        collection_name=vector_store.collection_name,
        ids=[other_document_chunk.enriched_chunk.chunk.chunk_id],
    )

    assert remaining == []
    assert len(other_remaining) == 1


async def test_delete_document_filters_by_version_when_given(vector_store):
    await vector_store.ensure_collection()
    chunk_v1 = make_embedded_chunk(document_id='fz-181', chunk_index=0)
    chunk_v2 = make_embedded_chunk(document_id='fz-181', chunk_index=1)
    metadata_v1 = DocumentMetadataInput(
        source_title='ФЗ-181', audience='both', topic='quota', version='2025-01-01', effective_date=date(2025, 1, 1)
    )
    metadata_v2 = DocumentMetadataInput(
        source_title='ФЗ-181', audience='both', topic='quota', version='2026-01-01', effective_date=date(2026, 1, 1)
    )

    await vector_store.upsert_chunk(chunk_v1, metadata_v1)
    await vector_store.upsert_chunk(chunk_v2, metadata_v2)

    await vector_store.delete_document('fz-181', version='2025-01-01')

    remaining_v1 = await vector_store.client.retrieve(
        collection_name=vector_store.collection_name, ids=[chunk_v1.enriched_chunk.chunk.chunk_id]
    )
    remaining_v2 = await vector_store.client.retrieve(
        collection_name=vector_store.collection_name, ids=[chunk_v2.enriched_chunk.chunk.chunk_id]
    )

    assert remaining_v1 == []
    assert len(remaining_v2) == 1
