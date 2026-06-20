from datetime import date

from qdrant_client import AsyncQdrantClient, models

from app.core.config_logger import logger
from app.models.metadata import ChunkMetadata
from app.models.schemas import DocumentMetadataInput, EmbeddedChunk

# Гипотетических вопросов на чанк — 3–5 (ChunkEnrichmentResult, Этап 3).
# Именованные векторы под вопросы заводятся в коллекции с запасом на
# максимум, не использованные слоты у конкретной точки просто не заполняются.
MAX_HYPOTHETICAL_QUESTIONS = 5
CHUNK_VECTOR_NAME = 'chunk'
QUESTION_VECTOR_NAMES = [f'question_{i}' for i in range(MAX_HYPOTHETICAL_QUESTIONS)]


# Поля payload, не входящие в формальную схему ChunkMetadata (раздел 3
# плана), но необходимые поисковому пайплайну: BM25 (Этап 5) считается по
# тексту чанка, а финальный ответ API (Этап 7) должен вернуть сам текст и
# заголовок, а не только метаданные для фильтрации.
TEXT_PAYLOAD_FIELD = 'text'
SYNTHETIC_TITLE_PAYLOAD_FIELD = 'synthetic_title'
HYPOTHETICAL_QUESTIONS_PAYLOAD_FIELD = 'hypothetical_questions'


def build_chunk_metadata(
    embedded_chunk: EmbeddedChunk, document_metadata: DocumentMetadataInput
) -> ChunkMetadata:
    """Собирает полную схему метаданных чанка (раздел 3 плана) из частей,
    вычисленных ingestion-пайплайном, и метаданных, заданных на уровне документа.

    Args:
        embedded_chunk: Чанк с векторами — результат Этапа 4.
        document_metadata: Метаданные документа, общие для всех его чанков.

    Returns:
        Полная схема метаданных чанка для payload точки в Qdrant.
    """
    chunk = embedded_chunk.enriched_chunk.chunk

    return ChunkMetadata(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        category=chunk.category,
        source_title=document_metadata.source_title,
        audience=document_metadata.audience,
        topic=document_metadata.topic,
        date_added=date.today(),
        chunk_index=chunk.chunk_index,
        version=document_metadata.version,
        effective_date=document_metadata.effective_date,
        is_active=True,
    )


def build_chunk_payload(
    embedded_chunk: EmbeddedChunk, document_metadata: DocumentMetadataInput
) -> dict:
    """Собирает полный payload точки Qdrant: метаданные + текст для поиска и выдачи.

    Args:
        embedded_chunk: Чанк с векторами — результат Этапа 4.
        document_metadata: Метаданные документа, общие для всех его чанков.

    Returns:
        Словарь payload, готовый для `PointStruct.payload`.
    """
    enriched_chunk = embedded_chunk.enriched_chunk
    metadata = build_chunk_metadata(embedded_chunk, document_metadata)

    return {
        **metadata.model_dump(mode='json'),
        TEXT_PAYLOAD_FIELD: enriched_chunk.chunk.text,
        SYNTHETIC_TITLE_PAYLOAD_FIELD: enriched_chunk.synthetic_title,
        HYPOTHETICAL_QUESTIONS_PAYLOAD_FIELD: enriched_chunk.hypothetical_questions,
    }


class QdrantVectorStore:
    """Обёртка над Qdrant: создание коллекции, upsert и удаление чанков (Этап 4).

    Один чанк — одна точка Qdrant с несколькими именованными векторами
    (основной вектор "заголовок+текст" + по одному вектору на каждый
    гипотетический вопрос) и единым payload — полной схемой метаданных
    чанка. Несколько точек на чанк не заводится: это избавляет от
    дедупликации по chunk_id на этапе fusion (Этап 5).
    """

    def __init__(self, client: AsyncQdrantClient, collection_name: str, vector_dim: int):
        self.client = client
        self.collection_name = collection_name
        self.vector_dim = vector_dim

    async def ensure_collection(self) -> None:
        """Создаёт коллекцию с нужной схемой именованных векторов, если её нет."""
        exists = await self.client.collection_exists(self.collection_name)
        if exists:
            return

        logger.info('💾 Создание коллекции Qdrant: %s', self.collection_name)
        vectors_config = {
            name: models.VectorParams(size=self.vector_dim, distance=models.Distance.COSINE)
            for name in [CHUNK_VECTOR_NAME, *QUESTION_VECTOR_NAMES]
        }
        await self.client.create_collection(
            collection_name=self.collection_name, vectors_config=vectors_config
        )
        logger.info('✅ Коллекция Qdrant создана: %s', self.collection_name)

    async def upsert_chunk(
        self, embedded_chunk: EmbeddedChunk, document_metadata: DocumentMetadataInput
    ) -> None:
        """Upsert одного чанка в Qdrant.

        Args:
            embedded_chunk: Чанк с векторами — результат Этапа 4.
            document_metadata: Метаданные документа, общие для всех его чанков.
        """
        await self.upsert_chunks([embedded_chunk], document_metadata)

    async def upsert_chunks(
        self, embedded_chunks: list[EmbeddedChunk], document_metadata: DocumentMetadataInput
    ) -> None:
        """Upsert батча чанков одного документа в Qdrant.

        Args:
            embedded_chunks: Чанки с векторами — результат Этапа 4.
            document_metadata: Метаданные документа, общие для всех его чанков.
        """
        points = []
        for embedded_chunk in embedded_chunks:
            chunk = embedded_chunk.enriched_chunk.chunk

            vector = {CHUNK_VECTOR_NAME: embedded_chunk.chunk_vector}
            for question_vector, vector_name in zip(
                embedded_chunk.question_vectors, QUESTION_VECTOR_NAMES, strict=False
            ):
                vector[vector_name] = question_vector

            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=build_chunk_payload(embedded_chunk, document_metadata),
                )
            )

        logger.info('💾 Upsert %d чанков в коллекцию %s.', len(points), self.collection_name)
        await self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info('✅ Upsert завершён: %d чанков.', len(points))

    async def delete_document(self, document_id: str, version: str | None = None) -> None:
        """Удаляет все чанки документа (опционально — только указанной версии).

        Используется явным workflow обновления документа (раздел 3, Этап 7
        плана): новая версия upsert'ится первой, старая удаляется по
        document_id+version только после успешного upsert новой.

        Args:
            document_id: Идентификатор документа.
            version: Если задано — удаляются только чанки этой версии.
        """
        conditions = [models.FieldCondition(key='document_id', match=models.MatchValue(value=document_id))]
        if version is not None:
            conditions.append(
                models.FieldCondition(key='version', match=models.MatchValue(value=version))
            )

        logger.info('🗑️ Удаление чанков документа %s (version=%s).', document_id, version)
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=models.Filter(must=conditions)),
        )
        logger.info('✅ Удаление завершено: %s.', document_id)

    async def get_document_versions(self, document_id: str) -> list[str]:
        """Возвращает версии, под которыми документ сейчас проиндексирован в Qdrant.

        Используется явным workflow обновления документа (раздел 3, Этап 7
        плана): перед upsert новой версии нужно знать, какие версии уже
        есть в индексе, чтобы удалить их после успешного upsert новой.

        Args:
            document_id: Идентификатор документа.

        Returns:
            Отсортированный список уникальных версий. Пустой список, если
            документ ещё не проиндексирован.
        """
        versions: set[str] = set()
        offset = None
        query_filter = models.Filter(
            must=[models.FieldCondition(key='document_id', match=models.MatchValue(value=document_id))]
        )

        while True:
            points, offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=['version'],
                with_vectors=False,
            )
            versions.update(point.payload['version'] for point in points)
            if offset is None:
                break

        return sorted(versions)

    async def list_chunks(self, document_id: str, version: str | None = None) -> list[dict]:
        """Возвращает payload всех чанков документа — для просмотра содержимого
        в админке (Этап 11 плана, расширение "просмотр того, что хранится").

        Args:
            document_id: Идентификатор документа.
            version: Если задано — только чанки этой версии.

        Returns:
            Payload чанков (включая `chunk_id`), отсортированные по `chunk_index`.
        """
        conditions = [models.FieldCondition(key='document_id', match=models.MatchValue(value=document_id))]
        if version is not None:
            conditions.append(
                models.FieldCondition(key='version', match=models.MatchValue(value=version))
            )
        query_filter = models.Filter(must=conditions)

        chunks: list[dict] = []
        offset = None
        while True:
            points, offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend({'chunk_id': str(point.id), **point.payload} for point in points)
            if offset is None:
                break

        return sorted(chunks, key=lambda chunk: chunk.get('chunk_index', 0))
