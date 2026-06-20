from typing import Annotated

from fastapi import Depends

from app.core.settings import get_settings
from app.vectorstore.client import qdrant_client
from app.vectorstore.qdrant_client import QdrantVectorStore


def get_vector_store() -> QdrantVectorStore:
    settings = get_settings()
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.qdrant.qdrant_collection,
        vector_dim=settings.yandex.yandex_embedding_dim,
    )


VectorStoreDep = Annotated[QdrantVectorStore, Depends(get_vector_store)]
