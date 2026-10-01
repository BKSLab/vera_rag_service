from datetime import date
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.clients.embeddings import EmbeddingClient
from app.clients.llm import LlmClient
from app.dependencies.auth import verify_api_key
from app.dependencies.services import get_ingestion_service
from app.exceptions.document import DocumentChangeLogRepositoryError
from app.exceptions.ingestion import IngestionIntegrityError
from app.main import app
from app.models.schemas import (
    Chunk,
    ChunkEnrichmentResult,
    DocumentMetadataInput,
    IngestRequest,
    SectionUpdateRequest,
)
from app.repositories.document import DocumentRepository
from app.repositories.document_change_log import DocumentChangeLogRepository
from app.services.ingestion import IngestionService
from app.vectorstore.qdrant_client import QdrantVectorStore


def _make_service() -> tuple[
    IngestionService,
    AsyncMock,
    AsyncMock,
    AsyncMock,
    AsyncMock,
]:
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = ChunkEnrichmentResult(
        synthetic_title='Заголовок',
        hypothetical_questions=['Вопрос 1?', 'Вопрос 2?', 'Вопрос 3?'],
    )
    embedding_client = AsyncMock(spec=EmbeddingClient)
    embedding_client.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
    vector_store = AsyncMock(spec=QdrantVectorStore)
    vector_store.get_document_versions.return_value = []
    vector_store.get_latest_revision_date.return_value = None
    document_repository = AsyncMock(spec=DocumentRepository)
    change_log_repository = AsyncMock(spec=DocumentChangeLogRepository)
    service = IngestionService(
        llm_client=llm_client,
        embedding_client=embedding_client,
        vector_store=vector_store,
        document_repository=document_repository,
        change_log_repository=change_log_repository,
        legal_sync_client=None,
    )
    return service, llm_client, embedding_client, vector_store, document_repository


def _metadata(version: str = '2026-01-01') -> DocumentMetadataInput:
    return DocumentMetadataInput(
        source_title='Источник',
        audience='both',
        topics=[],
        version=version,
        effective_date=date(2026, 1, 1),
    )


def _section_request(category: str = 'labor_code') -> SectionUpdateRequest:
    return SectionUpdateRequest(
        category=category,
        raw_text='1. Текст секции.',
        section_title='Статья 1',
        revision_date=date(2026, 1, 1),
        source_title='Источник',
        audience='both',
        topics=[],
    )


def _ingest_request(category: str = 'labor_code') -> IngestRequest:
    return IngestRequest(
        category=category,
        act_type='Кодекс Российской Федерации',
        act_number='197-ФЗ',
        act_date=date(2001, 12, 30),
        act_title='Трудовой кодекс Российской Федерации',
        source_title='Трудовой кодекс Российской Федерации',
        revision_date=date(2026, 1, 1),
        raw_text='Текст документа.',
        audience='both',
        topics=[],
    )


async def test_repeated_ingestion_of_same_version_is_idempotent():
    service, _, _, vector_store, document_repository = _make_service()
    stored_ids: set[str] = set()

    async def get_stored_ids(*args, **kwargs) -> list[str]:
        return sorted(stored_ids)

    async def upsert_chunks(embedded_chunks, document_metadata) -> None:
        stored_ids.update(
            embedded_chunk.enriched_chunk.chunk.chunk_id
            for embedded_chunk in embedded_chunks
        )

    async def count_chunks(*args, **kwargs) -> int:
        return len(stored_ids)

    vector_store.get_document_version_chunk_ids.side_effect = get_stored_ids
    vector_store.get_actual_document_chunk_ids.side_effect = get_stored_ids
    vector_store.upsert_chunks.side_effect = upsert_chunks
    vector_store.count_actual_document_chunks.side_effect = count_chunks

    first = await service.ingest_document(request=_ingest_request())
    first_ids = stored_ids.copy()
    second = await service.ingest_document(request=_ingest_request())

    assert stored_ids == first_ids
    assert len(stored_ids) == first.chunks_count == second.chunks_count
    assert document_repository.save_document.await_count == 2


async def test_duplicate_ids_fail_before_enrichment_and_registry_write(monkeypatch):
    service, llm_client, embedding_client, vector_store, document_repository = _make_service()
    duplicate_chunks = [
        Chunk(
            chunk_id='duplicate-id',
            chunk_index=index,
            chunk_number_in_section=index,
            document_id='doc-1',
            parent_id='doc-1:1',
            category='labor_code',
            section_index=0,
            section_number='1',
            section_title='Статья 1',
            text=f'Текст {index}',
        )
        for index in range(2)
    ]
    monkeypatch.setattr('app.services.ingestion.chunk_document', lambda sections, version: duplicate_chunks)

    with pytest.raises(IngestionIntegrityError) as raised:
        await service.ingest_document(request=_ingest_request())

    assert raised.value.duplicate_chunk_ids == ['duplicate-id']
    llm_client.get_llm_response.assert_not_awaited()
    embedding_client.get_embedding.assert_not_awaited()
    vector_store.upsert_chunks.assert_not_awaited()
    document_repository.save_document.assert_not_awaited()


async def test_document_post_upsert_integrity_failure_keeps_registry_and_old_version_untouched():
    service, _, _, vector_store, document_repository = _make_service()
    vector_store.get_document_versions.return_value = ['2025-01-01']
    vector_store.get_document_version_chunk_ids.return_value = []
    vector_store.count_actual_document_chunks.return_value = 1
    vector_store.get_actual_document_chunk_ids.return_value = ['unexpected-chunk-id']

    with pytest.raises(IngestionIntegrityError) as raised:
        await service.ingest_document(request=_ingest_request())

    assert raised.value.expected_count == 1
    assert raised.value.actual_count == 1
    assert raised.value.missing_chunk_ids
    assert raised.value.unexpected_chunk_ids == ['unexpected-chunk-id']
    vector_store.delete_chunks.assert_awaited_once_with(raised.value.missing_chunk_ids)
    document_repository.save_document.assert_not_awaited()
    vector_store.delete_document.assert_not_awaited()
    document_repository.mark_versions_inactive.assert_not_awaited()


async def test_section_post_upsert_integrity_failure_uses_parent_scope_and_keeps_old_chunks_actual():
    service, _, _, vector_store, _ = _make_service()
    vector_store.get_actual_section_chunk_ids.return_value = ['old-chunk-id']
    vector_store.get_section_version_chunk_ids.return_value = []
    vector_store.count_actual_section_chunks.return_value = 0
    vector_store.get_actual_section_version_chunk_ids.return_value = []

    with pytest.raises(IngestionIntegrityError) as raised:
        await service.ingest_section('doc-1', '1', _section_request())

    assert raised.value.scope == 'parent_id=doc-1:1'
    vector_store.count_actual_section_chunks.assert_awaited_once_with('doc-1:1', '2026-01-01')
    vector_store.get_actual_section_version_chunk_ids.assert_awaited_once_with(
        'doc-1:1', '2026-01-01'
    )
    vector_store.set_chunks_inactive.assert_not_awaited()


async def test_integrity_cleanup_deletes_only_ids_absent_before_current_run():
    service, _, _, vector_store, _ = _make_service()

    await service._cleanup_new_chunks(
        expected_chunk_ids={'existing-chunk-id', 'new-chunk-id'},
        preexisting_chunk_ids={'existing-chunk-id'},
    )

    vector_store.delete_chunks.assert_awaited_once_with(['new-chunk-id'])


def _make_section_upsert_consistent(vector_store) -> None:
    """Заставляет мок хранилища отражать то, что в него записали.

    Проверка целостности после upsert сравнивает ожидаемые идентификаторы
    чанков с фактическими, поэтому мок обязан возвращать именно те чанки,
    которые получил, иначе успешный сценарий недостижим.
    """
    stored: dict[str, set[str]] = {'chunk_ids': set()}

    async def remember(embedded_chunks, document_metadata):
        stored['chunk_ids'] = {
            embedded_chunk.enriched_chunk.chunk.chunk_id for embedded_chunk in embedded_chunks
        }

    vector_store.upsert_chunks.side_effect = remember
    vector_store.get_actual_section_chunk_ids.return_value = []
    vector_store.get_section_version_chunk_ids.return_value = []
    vector_store.count_actual_section_chunks.side_effect = (
        lambda parent_id, version: len(stored['chunk_ids'])
    )
    vector_store.get_actual_section_version_chunk_ids.side_effect = (
        lambda parent_id, version: list(stored['chunk_ids'])
    )
    vector_store.set_chunks_inactive.return_value = 0


async def test_put_section_for_other_npa_is_accepted():
    """Постановления Правительства меняются постатейно так же, как законы.

    Раньше `other_npa` отклонялась с 422. Ограничение снято вместе с
    подключением мониторинга постановлений в Legal Sync Service: без этого
    автоматические обновления постановлений упирались бы в отказ.
    """
    service, _, _, vector_store, _ = _make_service()
    _make_section_upsert_consistent(vector_store)
    previous_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[get_ingestion_service] = lambda: service
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url='http://test',
        ) as client:
            response = await client.put(
                '/api/v1/document/doc-1/sections/1',
                json={
                    'category': 'other_npa',
                    'raw_text': '1. Текст пункта.',
                    'section_title': 'Пункт 1',
                    'revision_date': '2026-01-01',
                    'source_title': 'Постановление',
                    'audience': 'both',
                    'topics': [],
                },
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)

    assert response.status_code == 200
    vector_store.upsert_chunks.assert_awaited()


async def test_put_section_for_case_law_still_returns_explicit_error():
    """У судебной практики нет устойчивой нумерации статей — только целый документ."""

    service, _, _, vector_store, _ = _make_service()
    previous_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[get_ingestion_service] = lambda: service
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url='http://test',
        ) as client:
            response = await client.put(
                '/api/v1/document/doc-1/sections/1',
                json={
                    'category': 'case_law',
                    'raw_text': '1. Текст пункта.',
                    'section_title': 'Пункт 1',
                    'revision_date': '2026-01-01',
                    'source_title': 'Постановление Пленума',
                    'audience': 'both',
                    'topics': [],
                },
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)

    assert response.status_code == 422
    assert "category='case_law'" in response.json()['detail']
    assert 'Гранулярное обновление не поддерживается' in response.json()['detail']
    vector_store.upsert_chunks.assert_not_awaited()


def _journal_entry(service: IngestionService):
    """Единственная запись, ушедшая в журнал изменений."""
    service.change_log_repository.add_entry.assert_awaited_once()
    return service.change_log_repository.add_entry.await_args.args[0]


async def test_applied_section_update_is_recorded_in_the_journal():
    service, _, _, vector_store, _ = _make_service()
    _make_section_upsert_consistent(vector_store)
    vector_store.set_chunks_inactive.return_value = 2

    await service.ingest_section('doc-1', '1', _section_request())

    entry = _journal_entry(service)
    assert entry.status == 'applied'
    assert entry.error is None
    assert entry.document_id == 'doc-1'
    assert entry.section_number == '1'
    assert entry.superseded_chunks == 2


async def test_rejected_section_update_is_recorded_and_error_is_reraised():
    """Отклонённая присылка должна оставлять след.

    Без записи неприменённое изменение выглядит в админке ровно так же, как
    отсутствие изменений: статья молча остаётся в прежней редакции.
    """
    service, _, _, vector_store, _ = _make_service()

    with pytest.raises(ValueError):
        await service.ingest_section('doc-1', '1', _section_request(category='case_law'))

    entry = _journal_entry(service)
    assert entry.status == 'rejected'
    assert 'case_law' in entry.error
    assert entry.chunks_count == 0
    vector_store.upsert_chunks.assert_not_awaited()


async def test_failed_section_update_is_recorded_separately_from_rejection():
    """Сорвавшаяся обработка отделена от отклонения: её имеет смысл повторить,
    а отклонённые данные будут отклонены снова.

    Ошибка берётся неожиданная (`RuntimeError`, не `LlmApiRequestError`)
    намеренно: журнал должен фиксировать и то, чего мы не предусмотрели, —
    иначе неизвестный сбой не оставит следа вообще.
    """
    service, llm_client, _, vector_store, _ = _make_service()
    _make_section_upsert_consistent(vector_store)
    llm_client.get_llm_response.side_effect = RuntimeError('LLM недоступен')

    with pytest.raises(RuntimeError):
        await service.ingest_section('doc-1', '1', _section_request())

    entry = _journal_entry(service)
    assert entry.status == 'failed'
    assert 'LLM недоступен' in entry.error


async def test_journal_failure_does_not_replace_the_real_error():
    """Сбой журнала не должен подменить собой настоящую причину отказа —
    иначе оператор будет чинить журнал вместо реальной проблемы."""
    service, _, _, _, _ = _make_service()
    service.change_log_repository.add_entry.side_effect = DocumentChangeLogRepositoryError('журнал недоступен')

    with pytest.raises(ValueError) as raised:
        await service.ingest_section('doc-1', '1', _section_request(category='case_law'))

    assert 'Гранулярное обновление не поддерживается' in str(raised.value)
