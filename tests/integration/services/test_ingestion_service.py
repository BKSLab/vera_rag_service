import asyncio
from datetime import date
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.clients.embeddings import EmbeddingClient
from app.clients.llm import LlmClient
from app.core.settings import get_settings
from app.db.models.document import Document
from app.models.schemas import ChunkEnrichmentResult, DocumentMetadataInput
from app.repositories.document import DocumentRepository
from app.services.ingestion import IngestionService
from app.vectorstore.qdrant_client import QdrantVectorStore

VECTOR_DIM = 4


@pytest_asyncio.fixture
async def vector_store():
    """Тестовая коллекция на реальном локальном Qdrant (docker-compose), с очисткой после теста."""
    settings = get_settings().qdrant
    client = AsyncQdrantClient(url=settings.qdrant_url)
    collection_name = f'test_{uuid4().hex}'
    store = QdrantVectorStore(client=client, collection_name=collection_name, vector_dim=VECTOR_DIM)
    await store.ensure_collection()

    yield store

    await client.delete_collection(collection_name)
    await client.close()


def make_ingestion_service(vector_store: QdrantVectorStore, db_session) -> IngestionService:
    """LLM/Embedding — моки (ingestion здесь не про реальные внешние API,
    а про идемпотентность/гонки на реальном Qdrant+Postgres, ING-1/ING-2/ING-3)."""
    fake_llm_client = AsyncMock(spec=LlmClient)
    fake_llm_client.get_llm_response.return_value = ChunkEnrichmentResult(
        synthetic_title='Заголовок', hypothetical_questions=['Вопрос 1?', 'Вопрос 2?', 'Вопрос 3?']
    )
    fake_embedding_client = AsyncMock(spec=EmbeddingClient)
    fake_embedding_client.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4]

    return IngestionService(
        llm_client=fake_llm_client,
        embedding_client=fake_embedding_client,
        vector_store=vector_store,
        document_repository=DocumentRepository(db_session),
    )


def make_document_metadata(version: str = '2026-01-01') -> DocumentMetadataInput:
    return DocumentMetadataInput(
        source_title='ФЗ-181, Статья 21',
        audience='both',
        topic='quota',
        version=version,
        effective_date=date(2026, 1, 1),
    )


async def test_ingest_document_does_not_duplicate_chunks_on_repeated_call_with_same_version(vector_store, db_session):
    """ING-1 — повторный ingest_document с тем же document_id+version не должен
    плодить дубликаты чанков в Qdrant (детерминированный chunk_id перезаписывает
    те же точки)."""
    service = make_ingestion_service(vector_store, db_session)
    document_id = f'doc-{uuid4().hex}'
    document_metadata = make_document_metadata()

    await service.ingest_document(
        document_id=document_id, raw_text='Текст документа про квоту.', category='federal_law',
        document_metadata=document_metadata,
    )
    first_chunks = await vector_store.list_chunks(document_id)

    await service.ingest_document(
        document_id=document_id, raw_text='Текст документа про квоту.', category='federal_law',
        document_metadata=document_metadata,
    )
    second_chunks = await vector_store.list_chunks(document_id)

    assert len(second_chunks) == len(first_chunks)
    assert {chunk['chunk_id'] for chunk in second_chunks} == {chunk['chunk_id'] for chunk in first_chunks}


async def test_ingest_document_concurrent_calls_same_document_same_version_do_not_duplicate(vector_store, engine):
    """ING-2 — двойной клик/повторный retry с одинаковыми параметрами (гонка)
    не должен оставлять дубликаты, даже при реальном параллельном выполнении.

    Каждый конкурентный вызов получает свою `AsyncSession` (как в проде —
    отдельная сессия на HTTP-запрос, `DbSessionDep`) — `AsyncSession` не
    безопасна для конкурентного использования из разных корутин одновременно,
    делить одну сессию между параллельными вызовами было бы нереалистичным
    тестовым сетапом, а не воспроизведением реальной гонки.
    """
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    document_id = f'doc-{uuid4().hex}'
    document_metadata = make_document_metadata()

    async def run() -> None:
        async with session_factory() as session:
            service = make_ingestion_service(vector_store, session)
            await service.ingest_document(
                document_id=document_id, raw_text='Текст документа про квоту.', category='federal_law',
                document_metadata=document_metadata,
            )

    await asyncio.gather(run(), run())

    chunks = await vector_store.list_chunks(document_id)
    chunk_ids = [chunk['chunk_id'] for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), 'не должно быть дублирующихся chunk_id'


async def test_ingest_document_concurrent_calls_different_versions_leave_exactly_one_active(vector_store, engine):
    """ING-2/ING-3 — конкурентные ingest_document для одного document_id, но
    разных версий, не должны оставить систему в противоречивом состоянии:
    ровно одна версия активна и в Postgres, и в Qdrant."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    document_id = f'doc-{uuid4().hex}'

    async def run(raw_text: str, version: str) -> None:
        async with session_factory() as session:
            service = make_ingestion_service(vector_store, session)
            await service.ingest_document(
                document_id=document_id, raw_text=raw_text, category='federal_law',
                document_metadata=make_document_metadata(version=version),
            )

    await asyncio.gather(
        run('Версия A документа.', '2026-01-01'),
        run('Версия B документа.', '2026-02-01'),
    )

    async with session_factory() as session:
        rows = (
            await session.execute(select(Document).where(Document.document_id == document_id))
        ).scalars().all()
    active_versions_in_registry = {row.version for row in rows if row.is_active}
    assert len(active_versions_in_registry) == 1

    chunks = await vector_store.list_chunks(document_id)
    versions_in_qdrant = {chunk['version'] for chunk in chunks}
    assert versions_in_qdrant == active_versions_in_registry
