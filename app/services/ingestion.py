from collections import Counter

from app.clients.embeddings import EmbeddingClient
from app.clients.llm import LlmClient
from app.core.config_logger import logger
from app.core.settings import get_settings
from app.db.models.document import Document
from app.embeddings.embedder import embed_chunks
from app.exceptions.document import DocumentRepositoryError
from app.exceptions.ingestion import (
    IngestionIntegrityError,
    RawTextTooLargeError,
    TooManyChunksError,
    TopicsNotAllowedForCategoryError,
)
from app.ingestion.chunking import chunk_document, compute_parent_id
from app.ingestion.enrichment import enrich_chunks, is_editorial_note_only
from app.ingestion.preprocess import preprocess_document
from app.models.metadata import Category
from app.models.schemas import (
    MAX_RAW_TEXT_LENGTH,
    SECTION_UPDATE_ALLOWED_CATEGORIES,
    TOPICS_ALLOWED_CATEGORIES,
    Chunk,
    DocumentMetadataInput,
    EmbeddedChunk,
    IngestResponse,
    Section,
    SectionUpdateRequest,
    SectionUpdateResponse,
)
from app.repositories.document import DocumentRepository
from app.vectorstore.qdrant_client import QdrantVectorStore

# Верхняя граница числа чанков одного документа (API-3,
# AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — без неё ingestion одного
# запроса мог бы запустить неограниченное число платных вызовов
# LLM-обогащения и эмбеддинга. С запасом от объёма ТК РФ (раздел 3.1 плана).
MAX_CHUNKS_PER_DOCUMENT = 2000


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
            RawTextTooLargeError: Если `raw_text` превышает `MAX_RAW_TEXT_LENGTH`.
            TopicsNotAllowedForCategoryError: Если заданы темы для category,
                где они не осмысленны (не в `TOPICS_ALLOWED_CATEGORIES`).
            TooManyChunksError: Если документ дал больше `MAX_CHUNKS_PER_DOCUMENT` чанков.
            LlmApiRequestError: Если обогащение хотя бы одного чанка не удалось.
            EmbeddingApiRequestError: Если эмбеддинг хотя бы одного чанка не удался.
        """
        if len(raw_text) > MAX_RAW_TEXT_LENGTH:
            raise RawTextTooLargeError(document_id, len(raw_text), MAX_RAW_TEXT_LENGTH)
        if document_metadata.topics and category not in TOPICS_ALLOWED_CATEGORIES:
            raise TopicsNotAllowedForCategoryError(document_id, category, document_metadata.topics)

        await self.document_repository.acquire_document_lock(document_id)
        try:
            old_versions = await self._find_old_versions(document_id, document_metadata.version)

            logger.info('🚀 Ingestion документа %s (version=%s).', document_id, document_metadata.version)
            embedded_chunks = await self._build_embedded_chunks(document_id, raw_text, category, document_metadata)

            expected_chunk_ids = self._embedded_chunk_ids(embedded_chunks)
            preexisting_chunk_ids = set(
                await self.vector_store.get_document_version_chunk_ids(
                    document_id, document_metadata.version
                )
            )
            await self.vector_store.upsert_chunks(embedded_chunks, document_metadata)
            await self._verify_document_integrity(
                document_id=document_id,
                version=document_metadata.version,
                expected_chunk_ids=expected_chunk_ids,
                preexisting_chunk_ids=preexisting_chunk_ids,
            )
            await self._save_document_record(document_id, category, document_metadata)
            not_removed_versions = await self._delete_old_versions(document_id, old_versions)

            logger.info(
                '✅ Ingestion документа %s завершён: %d чанков, заменено версий: %d.',
                document_id, len(embedded_chunks), len(old_versions),
            )
            return IngestResponse(
                document_id=document_id,
                version=document_metadata.version,
                chunks_count=len(embedded_chunks),
                replaced_versions=old_versions,
                not_removed_versions=not_removed_versions,
            )
        finally:
            await self.document_repository.release_document_lock(document_id)

    async def _find_old_versions(self, document_id: str, current_version: str) -> list[str]:
        """Версии документа, уже проиндексированные в Qdrant под другим `version`."""
        return [
            version
            for version in await self.vector_store.get_document_versions(document_id)
            if version != current_version
        ]

    async def _build_embedded_chunks(
        self, document_id: str, raw_text: str, category: Category, document_metadata: DocumentMetadataInput
    ) -> list[EmbeddedChunk]:
        """Препроцессинг → чанкинг → обогащение LLM → эмбеддинг (Этапы 1–4 плана).

        Raises:
            TooManyChunksError: Если документ дал больше `MAX_CHUNKS_PER_DOCUMENT` чанков.
        """
        sections = preprocess_document(document_id, raw_text, category)
        chunks = chunk_document(sections, version=document_metadata.version)
        self._ensure_unique_chunk_ids(
            document_id=document_id,
            version=document_metadata.version,
            scope='document',
            chunks=chunks,
        )
        chunks = [chunk for chunk in chunks if not is_editorial_note_only(chunk.text)]
        if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
            raise TooManyChunksError(document_id, len(chunks), MAX_CHUNKS_PER_DOCUMENT)
        enriched_chunks = await enrich_chunks(self.llm_client, chunks, document_metadata)
        return await embed_chunks(
            self.embedding_client, enriched_chunks, get_settings().yandex.embedding_doc_model_uri
        )

    async def _delete_old_versions(self, document_id: str, old_versions: list[str]) -> list[str]:
        """Удаляет чанки старых версий документа из Qdrant после успешного upsert новой.

        Отказ удаления одной версии не должен прерывать оставшиеся (ING-3) —
        новая версия к этому моменту уже полностью и успешно проиндексирована,
        падать с 500 здесь означало бы выдать клиенту ложное впечатление, что
        весь ingestion провалился, и спровоцировать повторный вызов (который
        упёрся бы в ING-1/ING-2). Версии, которые не удалось удалить,
        возвращаются явно — оператор может почистить их вручную через админку.

        Args:
            document_id: Идентификатор документа.
            old_versions: Версии, подлежащие удалению.

        Returns:
            Версии, удаление которых не удалось.
        """
        not_removed_versions: list[str] = []
        successfully_removed_versions: list[str] = []
        for old_version in old_versions:
            try:
                await self.vector_store.delete_document(document_id, version=old_version)
                successfully_removed_versions.append(old_version)
            except Exception as error:  # noqa: BLE001 — любой отказ конкретной версии не должен прервать остальные
                logger.warning(
                    '⚠️ Не удалось удалить версию %s документа %s. Детали: %s', old_version, document_id, error
                )
                not_removed_versions.append(old_version)

        await self._mark_old_versions_inactive(document_id, successfully_removed_versions)
        return not_removed_versions

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
                    topics=document_metadata.topics,
                    effective_date=document_metadata.effective_date,
                    is_active=True,
                )
            )
        except DocumentRepositoryError as error:
            logger.warning('⚠️ Не удалось записать реестр документа %s. Детали: %s', document_id, error)

    async def ingest_section(
        self,
        document_id: str,
        section_number: str,
        request: SectionUpdateRequest,
    ) -> SectionUpdateResponse:
        """Гранулярное обновление одной статьи/пункта (Этап 13 плана).

        Новые чанки upsert'ятся первыми (is_actual=True). Только после
        успешного завершения — старые чанки помечаются is_actual=False
        с effective_until = effective_date новой редакции. Физического
        удаления нет — история редакций хранится в Qdrant (для будущего
        запроса "на дату X").

        Args:
            document_id: Идентификатор документа.
            section_number: Номер статьи/пункта.
            request: Тело запроса с текстом, метаданными и версией.

        Returns:
            Сводка: количество новых чанков и сколько помечено неактуальными.

        Raises:
            ValueError: Если category не поддерживает гранулярное обновление.
            TopicsNotAllowedForCategoryError: Если заданы темы для category,
                где они не осмысленны (не в `TOPICS_ALLOWED_CATEGORIES`).
        """
        if request.category not in SECTION_UPDATE_ALLOWED_CATEGORIES:
            raise ValueError(
                f'Гранулярное обновление не поддерживается для category={request.category!r}. '
                f'Допустимые: {sorted(SECTION_UPDATE_ALLOWED_CATEGORIES)}.'
            )
        if request.topics and request.category not in TOPICS_ALLOWED_CATEGORIES:
            raise TopicsNotAllowedForCategoryError(document_id, request.category, request.topics)

        parent_id = compute_parent_id(document_id, section_number)

        section = Section(
            document_id=document_id,
            category=request.category,
            section_index=0,
            section_number=section_number,
            section_title=request.section_title,
            text=request.raw_text,
        )
        document_metadata = DocumentMetadataInput(
            source_title=request.source_title,
            audience=request.audience,
            topics=request.topics,
            version=request.version,
            effective_date=request.effective_date,
        )

        old_actual_chunk_ids = await self.vector_store.get_actual_section_chunk_ids(
            parent_id=parent_id,
            exclude_version=request.version,
        )

        chunks = chunk_document([section], version=request.version)
        self._ensure_unique_chunk_ids(
            document_id=document_id,
            version=request.version,
            scope=f'parent_id={parent_id}',
            chunks=chunks,
        )
        chunks = [chunk for chunk in chunks if not is_editorial_note_only(chunk.text)]
        enriched_chunks = await enrich_chunks(self.llm_client, chunks, document_metadata)
        embedded_chunks = await embed_chunks(
            self.embedding_client, enriched_chunks, get_settings().yandex.embedding_doc_model_uri
        )

        expected_chunk_ids = self._embedded_chunk_ids(embedded_chunks)
        preexisting_chunk_ids = set(
            await self.vector_store.get_section_version_chunk_ids(parent_id, request.version)
        )
        await self.vector_store.upsert_chunks(embedded_chunks, document_metadata)
        await self._verify_section_integrity(
            document_id=document_id,
            parent_id=parent_id,
            version=request.version,
            expected_chunk_ids=expected_chunk_ids,
            preexisting_chunk_ids=preexisting_chunk_ids,
        )

        superseded = await self.vector_store.set_chunks_inactive(
            chunk_ids=old_actual_chunk_ids,
            effective_until=request.effective_date,
        )

        logger.info(
            '✅ Секция %s документа %s обновлена: %d чанков, %d устарели.',
            section_number, document_id, len(embedded_chunks), superseded,
        )
        return SectionUpdateResponse(
            document_id=document_id,
            section_number=section_number,
            parent_id=parent_id,
            version=request.version,
            chunks_count=len(embedded_chunks),
            superseded_chunks=superseded,
        )

    @staticmethod
    def _embedded_chunk_ids(embedded_chunks: list[EmbeddedChunk]) -> set[str]:
        return {embedded_chunk.enriched_chunk.chunk.chunk_id for embedded_chunk in embedded_chunks}

    @staticmethod
    def _ensure_unique_chunk_ids(
        document_id: str,
        version: str,
        scope: str,
        chunks: list[Chunk],
    ) -> None:
        chunk_id_counts = Counter(chunk.chunk_id for chunk in chunks)
        duplicate_chunk_ids = [chunk_id for chunk_id, count in chunk_id_counts.items() if count > 1]
        if duplicate_chunk_ids:
            raise IngestionIntegrityError(
                document_id=document_id,
                version=version,
                scope=scope,
                expected_count=len(chunks),
                actual_count=len(chunk_id_counts),
                duplicate_chunk_ids=duplicate_chunk_ids,
            )

    async def _verify_document_integrity(
        self,
        document_id: str,
        version: str,
        expected_chunk_ids: set[str],
        preexisting_chunk_ids: set[str],
    ) -> None:
        actual_count = await self.vector_store.count_actual_document_chunks(document_id, version)
        actual_chunk_ids = set(await self.vector_store.get_actual_document_chunk_ids(document_id, version))
        if actual_count == len(expected_chunk_ids) and actual_chunk_ids == expected_chunk_ids:
            return

        await self._cleanup_new_chunks(expected_chunk_ids, preexisting_chunk_ids)
        raise IngestionIntegrityError(
            document_id=document_id,
            version=version,
            scope='document',
            expected_count=len(expected_chunk_ids),
            actual_count=actual_count,
            missing_chunk_ids=expected_chunk_ids - actual_chunk_ids,
            unexpected_chunk_ids=actual_chunk_ids - expected_chunk_ids,
        )

    async def _verify_section_integrity(
        self,
        document_id: str,
        parent_id: str,
        version: str,
        expected_chunk_ids: set[str],
        preexisting_chunk_ids: set[str],
    ) -> None:
        actual_count = await self.vector_store.count_actual_section_chunks(parent_id, version)
        actual_chunk_ids = set(
            await self.vector_store.get_actual_section_version_chunk_ids(parent_id, version)
        )
        if actual_count == len(expected_chunk_ids) and actual_chunk_ids == expected_chunk_ids:
            return

        await self._cleanup_new_chunks(expected_chunk_ids, preexisting_chunk_ids)
        raise IngestionIntegrityError(
            document_id=document_id,
            version=version,
            scope=f'parent_id={parent_id}',
            expected_count=len(expected_chunk_ids),
            actual_count=actual_count,
            missing_chunk_ids=expected_chunk_ids - actual_chunk_ids,
            unexpected_chunk_ids=actual_chunk_ids - expected_chunk_ids,
        )

    async def _cleanup_new_chunks(
        self,
        expected_chunk_ids: set[str],
        preexisting_chunk_ids: set[str],
    ) -> None:
        new_chunk_ids = sorted(expected_chunk_ids - preexisting_chunk_ids)
        if not new_chunk_ids:
            return
        try:
            await self.vector_store.delete_chunks(new_chunk_ids)
        except Exception as error:  # noqa: BLE001 — первичная integrity-ошибка важнее сбоя компенсации
            logger.exception(
                '❌ Не удалось удалить %d новых чанков после сбоя integrity-проверки: %s',
                len(new_chunk_ids),
                error,
            )

    async def _mark_old_versions_inactive(self, document_id: str, old_versions: list[str]) -> None:
        try:
            await self.document_repository.mark_versions_inactive(document_id, old_versions)
        except DocumentRepositoryError as error:
            logger.warning(
                '⚠️ Не удалось обновить реестр документа %s после удаления старых версий. Детали: %s',
                document_id, error,
            )
