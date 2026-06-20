from app.clients.embeddings import EmbeddingClient
from app.clients.llm import LlmClient
from app.core.config_logger import logger
from app.core.settings import get_settings
from app.db.models.document import Document
from app.embeddings.embedder import embed_chunks
from app.exceptions.document import DocumentRepositoryError
from app.ingestion.chunking import chunk_document
from app.ingestion.enrichment import enrich_chunks
from app.ingestion.preprocess import preprocess_document
from app.models.metadata import Category
from app.models.schemas import DocumentMetadataInput, IngestResponse
from app.repositories.document import DocumentRepository
from app.vectorstore.qdrant_client import QdrantVectorStore


class IngestionService:
    """Оркестратор ingestion-пайплайна для одного документа (Этапы 1–4, 7, 11.1)."""

    def __init__(
        self,
        llm_client: LlmClient,
        embedding_client: EmbeddingClient,
        vector_store: QdrantVectorStore,
        document_repository: DocumentRepository,
    ):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.document_repository = document_repository

    async def ingest_document(
        self,
        document_id: str,
        raw_text: str,
        category: Category,
        document_metadata: DocumentMetadataInput,
    ) -> IngestResponse:
        """Прогоняет документ через весь pipeline и upsert'ит его в Qdrant.

        Явный workflow обновления документа (раздел 3, Этап 7 плана): версии
        документа, уже проиндексированные под другим `version`, узнаются
        ДО upsert новой версии, но удаляются только ПОСЛЕ его успешного
        завершения — это гарантирует отсутствие окна недоступности источника.

        Args:
            document_id: Идентификатор документа.
            raw_text: Исходный текст документа.
            category: Категория источника (раздел 3 плана).
            document_metadata: Метаданные документа, общие для всех его чанков.

        Returns:
            Сводка ingestion: количество чанков и замещённые версии.

        Raises:
            LlmApiRequestError: Если обогащение хотя бы одного чанка не удалось.
            EmbeddingApiRequestError: Если эмбеддинг хотя бы одного чанка не удался.
        """
        old_versions = [
            version
            for version in await self.vector_store.get_document_versions(document_id)
            if version != document_metadata.version
        ]

        logger.info('🚀 Ingestion документа %s (version=%s).', document_id, document_metadata.version)
        sections = preprocess_document(document_id, raw_text, category)
        chunks = chunk_document(sections)
        enriched_chunks = await enrich_chunks(self.llm_client, chunks)
        embedded_chunks = await embed_chunks(
            self.embedding_client, enriched_chunks, get_settings().yandex.embedding_doc_model_uri
        )
        await self.vector_store.upsert_chunks(embedded_chunks, document_metadata)
        await self._save_document_record(document_id, category, document_metadata)

        for old_version in old_versions:
            await self.vector_store.delete_document(document_id, version=old_version)
        await self._mark_old_versions_inactive(document_id, old_versions)

        logger.info(
            '✅ Ingestion документа %s завершён: %d чанков, заменено версий: %d.',
            document_id, len(embedded_chunks), len(old_versions),
        )
        return IngestResponse(
            document_id=document_id,
            version=document_metadata.version,
            chunks_count=len(embedded_chunks),
            replaced_versions=old_versions,
        )

    async def _save_document_record(
        self, document_id: str, category: Category, document_metadata: DocumentMetadataInput
    ) -> None:
        """Пишет запись реестра документов (Этап 11.1). Источник правды о
        содержимом БЗ — Qdrant (upsert к этому моменту уже успешен), поэтому
        отказ записи сюда не должен ронять ingestion — перехватывается и
        логируется как предупреждение (FASTAPI_PATTERNS.md, раздел 9)."""
        try:
            await self.document_repository.save_document(
                Document(
                    document_id=document_id,
                    version=document_metadata.version,
                    category=category,
                    source_title=document_metadata.source_title,
                    audience=document_metadata.audience,
                    topic=document_metadata.topic,
                    effective_date=document_metadata.effective_date,
                    is_active=True,
                )
            )
        except DocumentRepositoryError as error:
            logger.warning('⚠️ Не удалось записать реестр документа %s. Детали: %s', document_id, error)

    async def _mark_old_versions_inactive(self, document_id: str, old_versions: list[str]) -> None:
        try:
            await self.document_repository.mark_versions_inactive(document_id, old_versions)
        except DocumentRepositoryError as error:
            logger.warning(
                '⚠️ Не удалось обновить реестр документа %s после удаления старых версий. Детали: %s',
                document_id, error,
            )
