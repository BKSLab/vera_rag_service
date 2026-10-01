from datetime import date
from unittest.mock import AsyncMock, MagicMock

from app.models.schemas import (
    Chunk,
    DocumentMetadataInput,
    EmbeddedChunk,
    EnrichedChunk,
    IngestRequest,
)
from app.services.ingestion import IngestionService


async def test_note_only_chunk_skips_enrichment_and_upsert_while_neighbor_is_indexed(monkeypatch):
    note_chunk = Chunk(
        chunk_id='note-only',
        chunk_index=0,
        chunk_number_in_section=0,
        document_id='fz-181',
        parent_id='fz-181:21',
        category='labor_code',
        section_index=0,
        section_number='21',
        section_title='Статья 21',
        text='(Статья 21 в редакции Федерального закона от 01.01.2026 № 1-ФЗ)',
    )
    content_chunk = Chunk(
        chunk_id='content',
        chunk_index=1,
        chunk_number_in_section=1,
        document_id='fz-181',
        parent_id='fz-181:21',
        category='labor_code',
        section_index=0,
        section_number='21',
        section_title='Статья 21',
        text='Работодатели обязаны соблюдать установленную квоту.',
    )
    enriched_content = EnrichedChunk(
        chunk=content_chunk,
        synthetic_title='Обязанность соблюдать квоту',
        hypothetical_questions=['Вопрос 1?', 'Вопрос 2?', 'Вопрос 3?'],
    )
    embedded_content = EmbeddedChunk(
        enriched_chunk=enriched_content,
        chunk_vector=[0.1, 0.2],
        question_vectors=[],
    )
    request = IngestRequest(
        category='labor_code',
        act_type='Кодекс Российской Федерации',
        act_number='197-ФЗ',
        act_date=date(2001, 12, 30),
        act_title='Трудовой кодекс Российской Федерации',
        source_title='Трудовой кодекс Российской Федерации',
        revision_date=date(2026, 1, 1),
        raw_text='Исходный текст.',
        audience='both',
        topics=[],
    )
    metadata = DocumentMetadataInput(
        source_title=request.source_title,
        audience=request.audience,
        topics=request.topics,
        version=request.version,
        effective_date=request.content_date,
    )
    llm_client = AsyncMock()
    embedding_client = AsyncMock()
    vector_store = AsyncMock()
    vector_store.get_document_versions.return_value = []
    vector_store.get_latest_revision_date.return_value = None
    vector_store.get_document_version_chunk_ids.return_value = []
    vector_store.count_actual_document_chunks.return_value = 1
    vector_store.get_actual_document_chunk_ids.return_value = ['content']
    document_repository = AsyncMock()
    document_repository.document_lock = MagicMock()
    enrich_mock = AsyncMock(return_value=[enriched_content])
    embed_mock = AsyncMock(return_value=[embedded_content])
    monkeypatch.setattr(
        'app.services.ingestion.chunk_document',
        lambda sections, version: [note_chunk, content_chunk],
    )
    monkeypatch.setattr('app.services.ingestion.enrich_chunks', enrich_mock)
    monkeypatch.setattr('app.services.ingestion.embed_chunks', embed_mock)
    service = IngestionService(
        llm_client=llm_client,
        embedding_client=embedding_client,
        vector_store=vector_store,
        document_repository=document_repository,
        change_log_repository=AsyncMock(),
        legal_sync_client=None,
    )

    await service.ingest_document(request=request)

    enrich_mock.assert_awaited_once_with(llm_client, [content_chunk], metadata)
    upserted_chunks = vector_store.upsert_chunks.await_args.args[0]
    assert [item.enriched_chunk.chunk.chunk_id for item in upserted_chunks] == ['content']
