from datetime import date
from uuid import uuid4

import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from app.core.settings import get_settings
from app.models.schemas import Chunk, DocumentMetadataInput, EmbeddedChunk, EnrichedChunk, SearchFilters
from app.search.hybrid import DENSE_TOP_K, dense_search, get_candidate_chunk_ids, hybrid_search, sparse_search
from app.vectorstore.qdrant_client import QdrantVectorStore

VECTOR_DIM = 4


@pytest_asyncio.fixture
async def populated_store():
    """Коллекция на реальном локальном Qdrant с тремя чанками разной аудитории/текста."""
    settings = get_settings().qdrant
    client = AsyncQdrantClient(url=settings.qdrant_url)
    collection_name = f'test_{uuid4().hex}'
    store = QdrantVectorStore(client=client, collection_name=collection_name, vector_dim=VECTOR_DIM)
    await store.ensure_collection()

    chunks_data = [
        ('Квота на трудоустройство инвалидов составляет от 2 до 4 процентов.', 'seeker', [1.0, 0.0, 0.0, 0.0]),
        ('Работодатель обязан создавать специальные рабочие места.', 'employer', [0.0, 1.0, 0.0, 0.0]),
        ('Общие положения трудового кодекса о праве на труд.', 'both', [0.0, 0.0, 1.0, 0.0]),
    ]
    embedded_chunks = []
    for text, audience, vector in chunks_data:
        chunk = Chunk(
            chunk_id=str(uuid4()), chunk_index=0, document_id='doc-1', category='labor_code',
            section_index=0, section_number=None, section_title='Секция', text=text,
        )
        enriched = EnrichedChunk(chunk=chunk, synthetic_title='Заголовок', hypothetical_questions=['В1?', 'В2?', 'В3?'])
        embedded_chunks.append(
            (EmbeddedChunk(enriched_chunk=enriched, chunk_vector=vector, question_vectors=[vector, vector, vector]), audience)
        )

    for embedded_chunk, audience in embedded_chunks:
        metadata = DocumentMetadataInput(
            source_title='Источник', audience=audience, topic='quota', version='2026-01-01', effective_date=date(2026, 1, 1)
        )
        await store.upsert_chunk(embedded_chunk, metadata)

    yield store, embedded_chunks

    await client.delete_collection(collection_name)
    await client.close()


async def test_dense_search_ranks_closest_vector_first(populated_store):
    store, embedded_chunks = populated_store
    seeker_chunk_id = embedded_chunks[0][0].enriched_chunk.chunk.chunk_id

    results = await dense_search(store.client, store.collection_name, query_vector=[1.0, 0.0, 0.0, 0.0])

    assert results[0][0] == seeker_chunk_id


async def test_dense_search_respects_audience_filter(populated_store):
    store, embedded_chunks = populated_store
    employer_chunk_id = embedded_chunks[1][0].enriched_chunk.chunk.chunk_id

    results = await dense_search(
        store.client, store.collection_name, query_vector=[0.0, 1.0, 0.0, 0.0],
        filters=SearchFilters(audience='seeker'),
    )

    result_ids = [chunk_id for chunk_id, _ in results]
    assert employer_chunk_id not in result_ids


async def test_dense_search_includes_both_audience_chunks_for_specific_audience_filter(populated_store):
    store, embedded_chunks = populated_store
    both_chunk_id = embedded_chunks[2][0].enriched_chunk.chunk.chunk_id

    results = await dense_search(
        store.client, store.collection_name, query_vector=[0.0, 0.0, 1.0, 0.0],
        filters=SearchFilters(audience='employer'),
    )

    result_ids = [chunk_id for chunk_id, _ in results]
    assert both_chunk_id in result_ids


async def test_sparse_search_finds_exact_term_match(populated_store):
    store, embedded_chunks = populated_store
    quota_chunk_id = embedded_chunks[0][0].enriched_chunk.chunk.chunk_id

    results = await sparse_search(store.client, store.collection_name, query_text='квота инвалидов процентов')

    assert results[0][0] == quota_chunk_id


async def test_sparse_search_returns_empty_list_when_filter_excludes_all_candidates(populated_store):
    store, _ = populated_store

    results = await sparse_search(
        store.client, store.collection_name, query_text='квота', filters=SearchFilters(topic='nonexistent')
    )

    assert results == []


async def test_get_candidate_chunk_ids_merges_dense_and_sparse_results(populated_store):
    store, embedded_chunks = populated_store
    quota_chunk_id = embedded_chunks[0][0].enriched_chunk.chunk.chunk_id

    candidates = await get_candidate_chunk_ids(
        store.client, store.collection_name, query_vector=[1.0, 0.0, 0.0, 0.0], query_text='квота инвалидов'
    )
    candidate_ids = [chunk_id for chunk_id, _ in candidates]

    assert quota_chunk_id in candidate_ids
    assert len(candidate_ids) == 3
    assert candidate_ids[0] == quota_chunk_id


@pytest_asyncio.fixture
async def imbalanced_category_store():
    """Коллекция, где `labor_code` крупный и почти идеально совпадает с
    запросом, а `case_law` — один чанк с заметно более низким cosine-score.
    Воспроизводит риск из раздела 4 плана: при плоском top-K `case_law`
    оказался бы за пределами DENSE_TOP_K и не дошёл бы до reranker'а."""
    settings = get_settings().qdrant
    client = AsyncQdrantClient(url=settings.qdrant_url)
    collection_name = f'test_{uuid4().hex}'
    store = QdrantVectorStore(client=client, collection_name=collection_name, vector_dim=VECTOR_DIM)
    await store.ensure_collection()

    metadata = DocumentMetadataInput(
        source_title='Источник', audience='both', topic='quota', version='2026-01-01', effective_date=date(2026, 1, 1)
    )

    labor_code_chunks = []
    for i in range(DENSE_TOP_K + 5):
        chunk = Chunk(
            chunk_id=str(uuid4()), chunk_index=i, document_id='tk-rf', category='labor_code',
            section_index=0, section_number=None, section_title='Секция', text=f'Норма ТК РФ номер {i}.',
        )
        enriched = EnrichedChunk(chunk=chunk, synthetic_title='Заголовок', hypothetical_questions=['В1?', 'В2?', 'В3?'])
        vector = [1.0, 0.0, 0.0, 0.0]
        embedded = EmbeddedChunk(enriched_chunk=enriched, chunk_vector=vector, question_vectors=[vector, vector, vector])
        labor_code_chunks.append(embedded)
        await store.upsert_chunk(embedded, metadata)

    case_law_chunk_text = 'Разъяснение Пленума ВС РФ по применению нормы.'
    case_law_chunk = Chunk(
        chunk_id=str(uuid4()), chunk_index=0, document_id='plenum-1', category='case_law',
        section_index=0, section_number=None, section_title='Секция', text=case_law_chunk_text,
    )
    case_law_enriched = EnrichedChunk(
        chunk=case_law_chunk, synthetic_title='Заголовок', hypothetical_questions=['В1?', 'В2?', 'В3?']
    )
    case_law_vector = [0.5, 0.5, 0.0, 0.0]
    case_law_embedded = EmbeddedChunk(
        enriched_chunk=case_law_enriched, chunk_vector=case_law_vector,
        question_vectors=[case_law_vector, case_law_vector, case_law_vector],
    )
    await store.upsert_chunk(case_law_embedded, metadata)

    yield store, case_law_embedded.enriched_chunk.chunk.chunk_id

    await client.delete_collection(collection_name)
    await client.close()


async def test_flat_dense_search_crowds_out_small_category(imbalanced_category_store):
    """Подтверждает саму проблему (раздел 4 плана) — без балансировки
    case_law не попадает даже в top-K плоского dense-поиска."""
    store, case_law_chunk_id = imbalanced_category_store

    results = await dense_search(store.client, store.collection_name, query_vector=[1.0, 0.0, 0.0, 0.0])

    result_ids = [chunk_id for chunk_id, _ in results]
    assert case_law_chunk_id not in result_ids


async def test_hybrid_search_balances_candidates_across_categories(imbalanced_category_store):
    """Этап 5.1 плана: без явного фильтра `category` hybrid_search ищет
    отдельно по каждой категории, поэтому case_law остаётся среди
    кандидатов для reranker'а, даже когда labor_code занял бы весь top-K."""
    store, case_law_chunk_id = imbalanced_category_store

    result = await hybrid_search(
        store.client, store.collection_name,
        query_vector=[1.0, 0.0, 0.0, 0.0], query_text='норма',
    )

    fused_ids = [chunk_id for chunk_id, _ in result.fused]
    assert case_law_chunk_id in fused_ids
