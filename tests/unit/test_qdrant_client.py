from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import Headers
from qdrant_client.http.exceptions import UnexpectedResponse

from app.vectorstore.qdrant_client import QdrantVectorStore


def _filter_values(query_filter) -> dict[str, object]:
    return {condition.key: condition.match.value for condition in query_filter.must}


async def test_ensure_collection_accepts_concurrent_creation_conflict():
    client = AsyncMock()
    client.collection_exists.return_value = False
    client.create_collection.side_effect = UnexpectedResponse(
        status_code=409,
        reason_phrase='Conflict',
        content=b'collection already exists',
        headers=Headers(),
    )
    store = QdrantVectorStore(client=client, collection_name='vera_kb', vector_dim=4)
    store._validate_existing_collection = AsyncMock()
    store._ensure_payload_indexes = AsyncMock()

    await store.ensure_collection()

    store._validate_existing_collection.assert_awaited_once()
    store._ensure_payload_indexes.assert_awaited_once()


async def test_ensure_collection_does_not_hide_other_qdrant_errors():
    client = AsyncMock()
    client.collection_exists.return_value = False
    error = UnexpectedResponse(
        status_code=500,
        reason_phrase='Internal Server Error',
        content=b'qdrant failed',
        headers=Headers(),
    )
    client.create_collection.side_effect = error
    store = QdrantVectorStore(client=client, collection_name='vera_kb', vector_dim=4)

    with pytest.raises(UnexpectedResponse) as raised:
        await store.ensure_collection()

    assert raised.value is error


async def test_get_actual_document_chunk_ids_scrolls_exact_document_version_scope():
    client = AsyncMock()
    client.scroll.side_effect = [
        ([SimpleNamespace(id='chunk-1')], 'next-page'),
        ([SimpleNamespace(id='chunk-2')], None),
    ]
    store = QdrantVectorStore(client=client, collection_name='vera_kb', vector_dim=4)

    chunk_ids = await store.get_actual_document_chunk_ids('doc-1', '2026-01-01')

    assert chunk_ids == ['chunk-1', 'chunk-2']
    query_filter = client.scroll.await_args_list[0].kwargs['scroll_filter']
    assert _filter_values(query_filter) == {
        'document_id': 'doc-1',
        'version': '2026-01-01',
        'is_actual': True,
    }


async def test_section_integrity_queries_use_parent_id_and_version_scope():
    client = AsyncMock()
    client.count.return_value = SimpleNamespace(count=2)
    client.scroll.return_value = ([SimpleNamespace(id='chunk-1'), SimpleNamespace(id='chunk-2')], None)
    store = QdrantVectorStore(client=client, collection_name='vera_kb', vector_dim=4)

    count = await store.count_actual_section_chunks('doc-1:1', '2026-01-01')
    chunk_ids = await store.get_actual_section_version_chunk_ids('doc-1:1', '2026-01-01')

    assert count == 2
    assert chunk_ids == ['chunk-1', 'chunk-2']
    count_filter = client.count.await_args.kwargs['count_filter']
    scroll_filter = client.scroll.await_args.kwargs['scroll_filter']
    expected_scope = {'parent_id': 'doc-1:1', 'version': '2026-01-01', 'is_actual': True}
    assert _filter_values(count_filter) == expected_scope
    assert _filter_values(scroll_filter) == expected_scope


async def test_pre_upsert_snapshots_include_inactive_points_in_exact_scope():
    client = AsyncMock()
    client.scroll.side_effect = [
        ([SimpleNamespace(id='document-chunk')], None),
        ([SimpleNamespace(id='section-chunk')], None),
    ]
    store = QdrantVectorStore(client=client, collection_name='vera_kb', vector_dim=4)

    document_ids = await store.get_document_version_chunk_ids('doc-1', '2026-01-01')
    section_ids = await store.get_section_version_chunk_ids('doc-1:1', '2026-01-01')

    assert document_ids == ['document-chunk']
    assert section_ids == ['section-chunk']
    document_filter = client.scroll.await_args_list[0].kwargs['scroll_filter']
    section_filter = client.scroll.await_args_list[1].kwargs['scroll_filter']
    assert _filter_values(document_filter) == {'document_id': 'doc-1', 'version': '2026-01-01'}
    assert _filter_values(section_filter) == {'parent_id': 'doc-1:1', 'version': '2026-01-01'}


async def test_delete_chunks_uses_exact_point_ids():
    client = AsyncMock()
    store = QdrantVectorStore(client=client, collection_name='vera_kb', vector_dim=4)

    await store.delete_chunks(['chunk-1', 'chunk-2'])

    selector = client.delete.await_args.kwargs['points_selector']
    assert selector.points == ['chunk-1', 'chunk-2']
