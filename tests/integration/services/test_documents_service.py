from datetime import date
from uuid import uuid4

import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select

from app.core.settings import get_settings
from app.db.models.document import Document
from app.models.schemas import Chunk, DocumentMetadataInput, EmbeddedChunk, EnrichedChunk
from app.repositories.document import DocumentRepository
from app.services.documents import DocumentsService
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


def make_embedded_chunk(document_id: str) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=str(uuid4()), chunk_index=0, document_id=document_id, category='labor_code',
        section_index=0, section_number='21', section_title='Статья 21', text='Текст чанка.',
    )
    enriched = EnrichedChunk(chunk=chunk, synthetic_title='Заголовок', hypothetical_questions=['В1?', 'В2?', 'В3?'])
    return EmbeddedChunk(enriched_chunk=enriched, chunk_vector=[0.1, 0.2, 0.3, 0.4], question_vectors=[])


async def test_delete_document_removes_chunks_and_registry_row(vector_store, db_session):
    """ARCH-4 — единая точка удаления: публичный API (как и админка) должен
    убирать и чанки из Qdrant, и строку реестра в Postgres, не только первое."""
    document_id = f'doc-{uuid4().hex}'
    document_repository = DocumentRepository(db_session)
    await document_repository.save_document(
        Document(
            document_id=document_id, version='2026-01-01', category='labor_code',
            source_title='Источник', audience='both', topic='quota',
            effective_date=date(2026, 1, 1), is_active=True,
        )
    )
    embedded_chunk = make_embedded_chunk(document_id)
    metadata = DocumentMetadataInput(
        source_title='Источник', audience='both', topic='quota', version='2026-01-01', effective_date=date(2026, 1, 1)
    )
    await vector_store.upsert_chunk(embedded_chunk, metadata)

    service = DocumentsService(vector_store=vector_store, document_repository=document_repository)
    await service.delete_document(document_id)

    remaining_chunks = await vector_store.list_chunks(document_id)
    assert remaining_chunks == []

    rows = (
        await db_session.execute(select(Document).where(Document.document_id == document_id))
    ).scalars().all()
    assert rows == []
