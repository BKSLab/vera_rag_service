from datetime import date

from qdrant_client import AsyncQdrantClient, models

from app.core.config_logger import logger
from app.models.metadata import ChunkMetadata
from app.models.schemas import DocumentMetadataInput, EmbeddedChunk
from app.vectorstore.sparse import SPARSE_VECTOR_NAME, text_to_sparse_vector

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

# Поля payload, по которым строятся фильтры почти в каждом запросе
# (`app/search/hybrid.py::build_qdrant_filter`, `delete_document`,
# `get_document_versions`, `list_chunks`) — без индекса Qdrant сканирует
# payload менее эффективно при росте коллекции (QD-1,
# AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md). Низкая кардинальность
# (`category` — 5 значений, `audience` — 3) — выигрыш от keyword-индекса
# особенно заметен.
# `parent_id` (keyword) — адресация секции при гранулярном обновлении (Этап 13).
# `is_actual` (bool) — дефолтный фильтр поиска, отсекающий исторические редакции.
PAYLOAD_INDEX_KEYWORD_FIELDS = ('category', 'audience', 'document_id', 'version', 'topic', 'parent_id')
PAYLOAD_INDEX_BOOL_FIELDS = ('is_actual',)


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
        parent_id=chunk.parent_id,
        category=chunk.category,
        source_title=document_metadata.source_title,
        audience=document_metadata.audience,
        topic=document_metadata.topic,
        date_added=date.today(),
        chunk_index=chunk.chunk_index,
        chunk_number_in_section=chunk.chunk_number_in_section,
        version=document_metadata.version,
        effective_date=document_metadata.effective_date,
        effective_until=None,
        is_actual=True,
        section_number=chunk.section_number,
        section_title=chunk.section_title,
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
        """Создаёт коллекцию с нужной схемой именованных векторов, sparse-вектором
        BM25 и payload-индексами, если коллекции ещё нет."""
        exists = await self.client.collection_exists(self.collection_name)
        if exists:
            return

        logger.info('💾 Создание коллекции Qdrant: %s', self.collection_name)
        # QD-2 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — int8-квантизация
        # только для основного вектора `chunk` (используется в каждом поисковом
        # запросе), не для вспомогательных `question_N` — ~4× экономия памяти
        # при потере recall на cosine обычно <1-2%. `always_ram=True`:
        # квантизованные векторы остаются в RAM даже если основной (полной
        # точности) вектор уйдёт на диск — поиск использует именно их.
        chunk_quantization_config = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
        )
        vectors_config = {
            name: models.VectorParams(
                size=self.vector_dim,
                distance=models.Distance.COSINE,
                quantization_config=chunk_quantization_config if name == CHUNK_VECTOR_NAME else None,
            )
            for name in [CHUNK_VECTOR_NAME, *QUESTION_VECTOR_NAMES]
        }
        # Нативный sparse-вектор с IDF-модификатором — BM25-подобное
        # ранжирование на стороне Qdrant (SEARCH-1/QD-3) вместо клиентского
        # `rank_bm25` поверх полного scroll коллекции на каждый запрос.
        sparse_vectors_config = {SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)}
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        for field_name in PAYLOAD_INDEX_KEYWORD_FIELDS:
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        for field_name in PAYLOAD_INDEX_BOOL_FIELDS:
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.BOOL,
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

            vector = {
                CHUNK_VECTOR_NAME: embedded_chunk.chunk_vector,
                SPARSE_VECTOR_NAME: text_to_sparse_vector(chunk.text),
            }
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

    async def set_section_inactive(self, parent_id: str, effective_until: date) -> int:
        """Помечает все актуальные чанки секции неактуальными (Этап 13 плана).

        Вызывается ПОСЛЕ успешного upsert новых чанков той же секции —
        старые чанки не удаляются физически, а получают is_actual=False
        и effective_until, чтобы юридически корректно отвечать на вопросы
        про события в прошлом ("какая норма действовала на дату X").

        Args:
            parent_id: Идентификатор секции (f"{document_id}:{section_number}").
            effective_until: Дата, когда эта редакция была заменена (= effective_date новой).

        Returns:
            Количество чанков, переведённых в неактуальное состояние.
        """
        query_filter = models.Filter(must=[
            models.FieldCondition(key='parent_id', match=models.MatchValue(value=parent_id)),
            models.FieldCondition(key='is_actual', match=models.MatchValue(value=True)),
        ])
        count_result = await self.client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
            exact=True,
        )
        superseded_count = count_result.count

        if superseded_count > 0:
            await self.client.set_payload(
                collection_name=self.collection_name,
                payload={'is_actual': False, 'effective_until': effective_until.isoformat()},
                points=models.FilterSelector(filter=query_filter),
            )
            logger.info(
                '📋 Секция %s: %d чанков помечены неактуальными (effective_until=%s).',
                parent_id, superseded_count, effective_until,
            )

        return superseded_count

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
