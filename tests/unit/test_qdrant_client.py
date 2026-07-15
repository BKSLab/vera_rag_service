from unittest.mock import AsyncMock

import pytest
from httpx import Headers
from qdrant_client.http.exceptions import UnexpectedResponse

from app.vectorstore.qdrant_client import QdrantVectorStore


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
