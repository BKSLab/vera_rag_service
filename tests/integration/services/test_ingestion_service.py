import asyncio
from datetime import date
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.clients.embeddings import EmbeddingClient
from app.clients.llm import LlmClient
from app.core.settings import get_settings
from app.db.models.document import Document
from app.exceptions.ingestion import TopicsNotAllowedForCategoryError
from app.models.schemas import ChunkEnrichmentResult, DocumentMetadataInput, SectionUpdateRequest
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
    # Пустые topics — тесты в этом файле используются с labor_code/federal_law
    # (идемпотентность/гонки ingestion, не про темы), а темы допустимы только
    # для other_npa/case_law/authorial (TopicsNotAllowedForCategoryError).
    return DocumentMetadataInput(
        source_title='ФЗ-181, Статья 21',
        audience='both',
        topics=[],
        version=version,
        effective_date=date(2026, 1, 1),
    )


def make_words(count: int, prefix: str) -> str:
    return ' '.join(f'{prefix}{index}' for index in range(count))


def make_section_update_request(raw_text: str, version: str) -> SectionUpdateRequest:
    return SectionUpdateRequest(
        category='labor_code',
        raw_text=raw_text,
        section_title='Статья 128. Отпуск без сохранения заработной платы',
        version=version,
        effective_date=date.fromisoformat(version),
        source_title='Трудовой кодекс Российской Федерации',
        audience='both',
        topics=[],
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


async def test_ingest_document_raises_when_topics_set_for_disallowed_category(vector_store, db_session):
    """Темы осмысленны только для other_npa/case_law/authorial (раздел 3
    плана, обсуждение с пользователем 2026-07-08) — labor_code/federal_law
    регулируют десятки тем одновременно, свести это к одной-двум означало
    бы соврать или обесценить фильтр."""
    service = make_ingestion_service(vector_store, db_session)
    document_metadata = DocumentMetadataInput(
        source_title='Источник', audience='both', topics=['квотирование'],
        version='2026-01-01', effective_date=date(2026, 1, 1),
    )

    with pytest.raises(TopicsNotAllowedForCategoryError):
        await service.ingest_document(
            document_id=f'doc-{uuid4().hex}', raw_text='Текст документа.', category='federal_law',
            document_metadata=document_metadata,
        )


async def test_ingest_document_allows_topics_for_other_npa(vector_store, db_session):
    service = make_ingestion_service(vector_store, db_session)
    document_id = f'doc-{uuid4().hex}'
    document_metadata = DocumentMetadataInput(
        source_title='Источник', audience='both', topics=['квотирование', 'трудоустройство'],
        version='2026-01-01', effective_date=date(2026, 1, 1),
    )

    await service.ingest_document(
        document_id=document_id, raw_text='Текст постановления.', category='other_npa',
        document_metadata=document_metadata,
    )

    chunks = await vector_store.list_chunks(document_id)
    assert chunks
    assert chunks[0]['topics'] == ['квотирование', 'трудоустройство']


async def test_ingest_section_raises_when_topics_set_for_disallowed_category(vector_store, db_session):
    """Гранулярное обновление секции подчиняется тому же правилу — темы
    допустимы только для other_npa из SECTION_UPDATE_ALLOWED_CATEGORIES
    (labor_code/federal_law поддерживают гранулярное обновление, но не темы)."""
    service = make_ingestion_service(vector_store, db_session)
    request = SectionUpdateRequest(
        category='labor_code',
        raw_text=make_words(20, 'слово'),
        section_title='Статья 128. Отпуск без сохранения заработной платы',
        version='2026-01-01',
        effective_date=date(2026, 1, 1),
        source_title='ТК РФ',
        audience='both',
        topics=['увольнение'],
    )

    with pytest.raises(TopicsNotAllowedForCategoryError):
        await service.ingest_section(f'doc-{uuid4().hex}', '128', request)


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


async def test_ingest_section_keeps_new_revision_actual_when_chunk_count_grows(vector_store, db_session):
    """Гранулярное обновление статьи должно помечать неактуальными только
    старые point IDs, найденные до upsert новой редакции. Если старая
    редакция была одним чанком, а новая стала несколькими, новые чанки
    обязаны остаться is_actual=True и быть доступными поиску."""
    service = make_ingestion_service(vector_store, db_session)
    document_id = f'tk-rf-{uuid4().hex}'
    section_number = '128'

    await service.ingest_section(
        document_id=document_id,
        section_number=section_number,
        request=make_section_update_request(
            raw_text='Работнику может быть предоставлен отпуск без сохранения заработной платы.',
            version='2026-01-01',
        ),
    )

    await service.ingest_section(
        document_id=document_id,
        section_number=section_number,
        request=make_section_update_request(
            raw_text='\n'.join([
                make_words(220, 'редакцияА'),
                make_words(220, 'редакцияБ'),
            ]),
            version='2026-02-01',
        ),
    )

    chunks = await vector_store.list_chunks(document_id)
    old_chunks = [chunk for chunk in chunks if chunk['version'] == '2026-01-01']
    new_chunks = [chunk for chunk in chunks if chunk['version'] == '2026-02-01']

    assert len(old_chunks) == 1
    assert len(new_chunks) > 1
    assert all(chunk['is_actual'] is False for chunk in old_chunks)
    assert all(chunk['effective_until'] == '2026-02-01' for chunk in old_chunks)
    assert all(chunk['is_actual'] is True for chunk in new_chunks)
    assert all(chunk['effective_until'] is None for chunk in new_chunks)


async def test_ingest_section_keeps_new_revision_actual_when_chunk_count_shrinks(vector_store, db_session):
    """Обратный сценарий: старая редакция секции была несколькими чанками,
    новая стала одним. Все старые чанки должны стать historical, а новый
    единственный чанк должен остаться actual."""
    service = make_ingestion_service(vector_store, db_session)
    document_id = f'tk-rf-{uuid4().hex}'
    section_number = '129'

    await service.ingest_section(
        document_id=document_id,
        section_number=section_number,
        request=make_section_update_request(
            raw_text='\n'.join([
                make_words(220, 'стараяА'),
                make_words(220, 'стараяБ'),
            ]),
            version='2026-01-01',
        ),
    )

    await service.ingest_section(
        document_id=document_id,
        section_number=section_number,
        request=make_section_update_request(
            raw_text='Новая краткая редакция статьи.',
            version='2026-02-01',
        ),
    )

    chunks = await vector_store.list_chunks(document_id)
    old_chunks = [chunk for chunk in chunks if chunk['version'] == '2026-01-01']
    new_chunks = [chunk for chunk in chunks if chunk['version'] == '2026-02-01']

    assert len(old_chunks) > 1
    assert len(new_chunks) == 1
    assert all(chunk['is_actual'] is False for chunk in old_chunks)
    assert all(chunk['effective_until'] == '2026-02-01' for chunk in old_chunks)
    assert new_chunks[0]['is_actual'] is True
    assert new_chunks[0]['effective_until'] is None
