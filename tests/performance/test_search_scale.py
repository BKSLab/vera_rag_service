import time
from datetime import date
from uuid import uuid4

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from app.core.settings import get_settings
from app.models.schemas import Chunk, DocumentMetadataInput, EmbeddedChunk, EnrichedChunk
from app.search.hybrid import dense_search, sparse_search
from app.vectorstore.qdrant_client import QdrantVectorStore

# TEST-3 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — регрессионный
# тест именно на находку SEARCH-1/QD-3: раньше `sparse_search` выгружала
# весь корпус (`scroll`) и пересчитывала `rank_bm25.BM25Okapi` с нуля на
# каждый запрос — O(N) от размера корпуса. Нативные sparse-векторы Qdrant
# (IDF) превращают это в обычный индексный запрос — latency не должна
# заметно расти с размером корпуса. Не входит в обычный прогон (`pytest.ini`
# — `addopts = -m "not slow"`), запускать явно: `pytest -m slow tests/performance`.
CORPUS_SIZE = 5000
LATENCY_THRESHOLD_SECONDS = 2.0
VECTOR_DIM = 4
UPSERT_BATCH_SIZE = 500


def make_embedded_chunk(index: int) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=str(uuid4()), chunk_index=index, chunk_number_in_section=0,
        document_id='tk-rf', parent_id=f'tk-rf:{index}', category='labor_code',
        section_index=0, section_number=str(index), section_title=f'Статья {index}',
        text=f'Норма трудового кодекса номер {index} о трудоустройстве и правах работников.',
    )
    enriched = EnrichedChunk(chunk=chunk, synthetic_title='Заголовок', hypothetical_questions=['В1?', 'В2?', 'В3?'])
    vector = [float(index % 7), float((index + 1) % 5), float((index + 2) % 3), 0.0]
    return EmbeddedChunk(enriched_chunk=enriched, chunk_vector=vector, question_vectors=[])


@pytest_asyncio.fixture
async def large_corpus_store():
    settings = get_settings().qdrant
    client = AsyncQdrantClient(url=settings.qdrant_url)
    collection_name = f'test_perf_{uuid4().hex}'
    store = QdrantVectorStore(client=client, collection_name=collection_name, vector_dim=VECTOR_DIM)
    await store.ensure_collection()

    metadata = DocumentMetadataInput(
        source_title='ТК РФ', audience='both', topic='quota', version='2026-01-01', effective_date=date(2026, 1, 1)
    )
    batch: list[EmbeddedChunk] = []
    for index in range(CORPUS_SIZE):
        batch.append(make_embedded_chunk(index))
        if len(batch) >= UPSERT_BATCH_SIZE:
            await store.upsert_chunks(batch, metadata)
            batch = []
    if batch:
        await store.upsert_chunks(batch, metadata)

    yield store

    await client.delete_collection(collection_name)
    await client.close()


@pytest.mark.slow
async def test_sparse_search_latency_does_not_grow_with_corpus_size(large_corpus_store):
    started_at = time.perf_counter()
    results = await sparse_search(
        large_corpus_store.client, large_corpus_store.collection_name, query_text='норма трудоустройства'
    )
    elapsed = time.perf_counter() - started_at

    assert elapsed < LATENCY_THRESHOLD_SECONDS, (
        f'sparse_search занял {elapsed:.2f}с на корпусе {CORPUS_SIZE} чанков — '
        f'похоже на регрессию SEARCH-1 (полный scroll+BM25 вместо нативного sparse-вектора).'
    )
    assert len(results) > 0


@pytest.mark.slow
async def test_dense_search_latency_does_not_grow_with_corpus_size(large_corpus_store):
    started_at = time.perf_counter()
    results = await dense_search(
        large_corpus_store.client, large_corpus_store.collection_name, query_vector=[1.0, 1.0, 1.0, 0.0]
    )
    elapsed = time.perf_counter() - started_at

    assert elapsed < LATENCY_THRESHOLD_SECONDS
    assert len(results) > 0
