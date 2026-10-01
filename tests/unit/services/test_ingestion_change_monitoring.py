from datetime import date
from unittest.mock import AsyncMock

import pytest

from app.clients.embeddings import EmbeddingClient
from app.clients.legal_sync import LegalSyncClient
from app.clients.llm import LlmClient
from app.core.settings import LegalSyncSettings, Settings, get_settings
from app.exceptions.document import DocumentChangeLogRepositoryError
from app.exceptions.legal_sync import LegalSyncClientError
from app.models.schemas import ChunkEnrichmentResult, IngestRequest, SectionUpdateRequest
from app.repositories.document import DocumentRepository
from app.repositories.document_change_log import DocumentChangeLogRepository
from app.services.ingestion import IngestionService
from app.vectorstore.qdrant_client import QdrantVectorStore


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _enable_monitoring(monkeypatch, **overrides) -> None:
    """Включает мониторинг изменений в настройках на время теста."""

    values = {'legal_sync_enabled': True, 'legal_sync_base_url': 'http://legal-sync.test'}
    values.update(overrides)
    settings = get_settings()
    patched = Settings.model_construct(
        **{**settings.__dict__, 'legal_sync': LegalSyncSettings(**values)}
    )
    monkeypatch.setattr('app.services.ingestion.get_settings', lambda: patched)


def _make_service(legal_sync_client=None):
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
    stored: dict[str, set[str]] = {'chunk_ids': set()}

    async def remember(embedded_chunks, document_metadata):
        stored['chunk_ids'] = {
            embedded_chunk.enriched_chunk.chunk.chunk_id for embedded_chunk in embedded_chunks
        }

    vector_store.upsert_chunks.side_effect = remember
    vector_store.get_document_version_chunk_ids.return_value = []
    vector_store.count_actual_document_chunks.side_effect = (
        lambda document_id, version: len(stored['chunk_ids'])
    )
    vector_store.get_actual_document_chunk_ids.side_effect = (
        lambda document_id, version: list(stored['chunk_ids'])
    )
    vector_store.get_actual_section_chunk_ids.return_value = []
    vector_store.get_section_version_chunk_ids.return_value = []
    vector_store.count_actual_section_chunks.side_effect = (
        lambda parent_id, version: len(stored['chunk_ids'])
    )
    vector_store.get_actual_section_version_chunk_ids.side_effect = (
        lambda parent_id, version: list(stored['chunk_ids'])
    )
    vector_store.set_chunks_inactive.return_value = 2

    change_log_repository = AsyncMock(spec=DocumentChangeLogRepository)
    service = IngestionService(
        llm_client=llm_client,
        embedding_client=embedding_client,
        vector_store=vector_store,
        document_repository=AsyncMock(spec=DocumentRepository),
        change_log_repository=change_log_repository,
        legal_sync_client=legal_sync_client,
    )
    return service, change_log_repository, vector_store


def _ingest_request(category: str = 'federal_law') -> IngestRequest:
    return IngestRequest(
        category=category,
        act_type='Федеральный закон' if category != 'case_law' else 'Обзор судебной практики',
        act_number='181-ФЗ' if category != 'case_law' else '1',
        act_date=date(1995, 11, 24),
        act_title='О социальной защите инвалидов в Российской Федерации',
        source_title='Федеральный закон «О социальной защите инвалидов в Российской Федерации» от 24 ноября 1995 N 181-ФЗ',
        revision_date=date(2026, 1, 1) if category in {'federal_law', 'labor_code', 'other_npa'} else None,
        raw_text='Статья 1. Общие положения\nСодержательный текст статьи.',
        audience='both',
        topics=[],
    )


def _section_request() -> SectionUpdateRequest:
    return SectionUpdateRequest(
        category='labor_code',
        raw_text='Статья 59. Срочный трудовой договор заключается...',
        section_title='Срочный трудовой договор',
        revision_date=date(2027, 3, 1),
        source_title='"Трудовой кодекс Российской Федерации" от 30.12.2001 № 197-ФЗ',
        audience='both',
        topics=[],
        amending_act_type='Федеральный закон',
        amending_act_number='246-ФЗ',
        amending_act_date=date(2026, 7, 26),
    )


# Блок журнала изменений


async def test_section_update_is_recorded_in_change_log():
    service, change_log_repository, _ = _make_service()

    await service.ingest_section('labor_code_rf', '59', _section_request())

    change_log_repository.add_entry.assert_awaited_once()
    entry = change_log_repository.add_entry.await_args.args[0]
    assert entry.document_id == 'labor_code_rf'
    assert entry.section_number == '59'
    assert entry.revision_date == date(2027, 3, 1)
    assert entry.amending_act_number == '246-ФЗ'
    assert entry.superseded_chunks == 2


async def test_change_log_failure_does_not_break_section_update():
    """Переиндексация уже выполнена — отказ журнала не может её отменить."""

    service, change_log_repository, _ = _make_service()
    change_log_repository.add_entry.side_effect = DocumentChangeLogRepositoryError('БД недоступна')

    result = await service.ingest_section('labor_code_rf', '59', _section_request())

    assert result.section_number == '59'


# Блок постановки документа на контроль изменений


async def test_ingest_registers_document_for_monitoring(monkeypatch):
    _enable_monitoring(monkeypatch)
    legal_sync_client = AsyncMock(spec=LegalSyncClient)
    service, _, _ = _make_service(legal_sync_client=legal_sync_client)

    await service.ingest_document(request=_ingest_request())

    legal_sync_client.register_tracked_document.assert_awaited_once()
    assert legal_sync_client.register_tracked_document.await_args.kwargs['document_id'] == 'fz-181-1995'


async def test_untracked_category_is_not_registered(monkeypatch):
    _enable_monitoring(monkeypatch)
    legal_sync_client = AsyncMock(spec=LegalSyncClient)
    service, _, _ = _make_service(legal_sync_client=legal_sync_client)

    await service.ingest_document(request=_ingest_request(category='case_law'))

    legal_sync_client.register_tracked_document.assert_not_awaited()


async def test_monitoring_disabled_skips_registration(monkeypatch):
    _enable_monitoring(monkeypatch, legal_sync_enabled=False)
    legal_sync_client = AsyncMock(spec=LegalSyncClient)
    service, _, _ = _make_service(legal_sync_client=legal_sync_client)

    await service.ingest_document(request=_ingest_request())

    legal_sync_client.register_tracked_document.assert_not_awaited()


async def test_monitoring_failure_does_not_break_ingestion(monkeypatch):
    """Документ уже проиндексирован — недоступность мониторинга не отменяет ingest."""

    _enable_monitoring(monkeypatch)
    legal_sync_client = AsyncMock(spec=LegalSyncClient)
    legal_sync_client.register_tracked_document.side_effect = LegalSyncClientError('недоступен')
    service, _, _ = _make_service(legal_sync_client=legal_sync_client)

    result = await service.ingest_document(request=_ingest_request())

    assert result.document_id == 'fz-181-1995'
    assert result.version == '2026-01-01'
