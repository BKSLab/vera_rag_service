from app.vectorstore.qdrant_client import QdrantVectorStore


class DocumentsService:
    """Управление документами в базе знаний на уровне всего документа (Этап 7)."""

    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store

    async def delete_document(self, document_id: str) -> None:
        """Удаляет документ и все его чанки (все версии) из Qdrant.

        Args:
            document_id: Идентификатор документа.
        """
        await self.vector_store.delete_document(document_id)
