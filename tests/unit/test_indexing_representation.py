from datetime import date
from unittest.mock import AsyncMock

from qdrant_client import models

from app.models.schemas import Chunk, DocumentMetadataInput, EmbeddedChunk, EnrichedChunk
from app.vectorstore.qdrant_client import QdrantVectorStore


async def test_upsert_indexes_prefix_and_source_title_without_changing_chunk_text(monkeypatch):
    original_text = 'Работнику предоставляется отпуск без сохранения заработной платы.'
    chunk = Chunk(
        chunk_id='chunk-128',
        chunk_index=0,
        chunk_number_in_section=0,
        document_id='tk-rf',
        parent_id='tk-rf:128',
        category='labor_code',
        section_index=0,
        section_number='128',
        section_title='Статья 128. Отпуск без сохранения заработной платы',
        text=original_text,
    )
    embedded_chunk = EmbeddedChunk(
        enriched_chunk=EnrichedChunk(
            chunk=chunk,
            synthetic_title='Отпуск за свой счёт',
            hypothetical_questions=['Вопрос 1?', 'Вопрос 2?', 'Вопрос 3?'],
        ),
        chunk_vector=[0.1, 0.2],
        question_vectors=[],
    )
    metadata = DocumentMetadataInput(
        source_title='ТК РФ',
        audience='both',
        topics=[],
        version='2026-01-01',
        effective_date=date(2026, 1, 1),
    )
    indexed_texts: list[str] = []

    def capture_sparse_text(text: str) -> models.SparseVector:
        indexed_texts.append(text)
        return models.SparseVector(indices=[], values=[])

    monkeypatch.setattr('app.vectorstore.qdrant_client.text_to_sparse_vector', capture_sparse_text)
    client = AsyncMock()
    store = QdrantVectorStore(client=client, collection_name='vera_kb', vector_dim=2)

    await store.upsert_chunks([embedded_chunk], metadata)

    assert indexed_texts == [
        'Статья 128. Отпуск без сохранения заработной платы\nТК РФ\n'
        'Работнику предоставляется отпуск без сохранения заработной платы.'
    ]
    assert chunk.text == original_text
    point = client.upsert.await_args.kwargs['points'][0]
    assert point.payload['text'] == original_text
