from collections import Counter
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import date

from app.clients.embeddings import EmbeddingClient
from app.clients.legal_sync import LegalSyncClient
from app.clients.llm import LlmClient
from app.core.config_logger import logger
from app.core.settings import get_settings
from app.db.models.document import Document
from app.db.models.document_change_log import (
    CHANGE_APPLIED,
    CHANGE_FAILED,
    CHANGE_REJECTED,
    DocumentChangeLog,
)
from app.embeddings.embedder import embed_chunks
from app.exceptions.document import DocumentChangeLogRepositoryError, DocumentRepositoryError
from app.exceptions.ingestion import (
    EmptyIngestionContentError,
    IngestionIntegrityError,
    RawTextTooLargeError,
    StaleRevisionError,
    TooManyChunksError,
    TopicsNotAllowedForCategoryError,
)
from app.exceptions.legal_sync import LegalSyncClientError
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
    IngestRequest,
    IngestResponse,
    Section,
    SectionUpdateRequest,
    SectionUpdateResponse,
)
from app.repositories.document import DocumentRepository
from app.repositories.document_change_log import DocumentChangeLogRepository
from app.repositories.ingestion_run import IngestionRunRepository
from app.services.ingestion_journal import IngestionTrace, current_trace, ingestion_stage, stage_warning
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
        change_log_repository: DocumentChangeLogRepository,
        legal_sync_client: LegalSyncClient | None,
        ingestion_run_repository: IngestionRunRepository | None = None,
        source: str = 'api',
        change_id: int | None = None,
    ):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.document_repository = document_repository
        self.change_log_repository = change_log_repository
        # Клиент отсутствует, когда мониторинг изменений выключен настройкой:
        # тогда постановка на контроль просто не выполняется.
        self.legal_sync_client = legal_sync_client
        self.ingestion_run_repository = ingestion_run_repository
        self.source = source
        self.change_id = change_id

    async def ingest_document(self, request: IngestRequest) -> IngestResponse:
        trace = self._trace(request, request.document_id)
        async with trace.run():
            result = await self._ingest_document(request)
            trace.values['result'].update(result.model_dump(mode='json', exclude_unset=True))
        return result.model_copy(update=self._receipt(trace))

    @staticmethod
    def _receipt(trace: IngestionTrace) -> dict:
        return {
            'operation_id': trace.values['id'], 'status': trace.values['status'], 'warnings': trace.warnings,
            'collection_name': trace.values['collection_name'], 'input_sha256': trace.values['result']['input_sha256'],
            'integrity_verified': trace.steps['integrity'].data['status'] == 'succeeded',
        }

    def _trace(self, request, document_id: str, section_number: str | None = None) -> IngestionTrace:
        return IngestionTrace(
            self.ingestion_run_repository, raw_text=request.raw_text,
            document_id=document_id, source_title=request.source_title,
            kind='section' if section_number is not None else 'document', source=self.source,
            version=request.version, revision_date=request.revision_date if section_number is not None else request.content_date,
            section_number=section_number, collection_name=getattr(self.vector_store, 'collection_name', get_settings().qdrant.qdrant_collection),
            change_id=self.change_id,
        )

    async def _ingest_document(self, request: IngestRequest) -> IngestResponse:
        """Прогоняет документ через весь pipeline и upsert'ит его в Qdrant.

        Явный workflow обновления документа (раздел 3, Этап 7 плана): версии
        документа, уже проиндексированные под другим `version`, узнаются
        ДО upsert новой версии, но удаляются только ПОСЛЕ его успешного
        завершения — это гарантирует отсутствие окна недоступности источника.

        Запрос принимается целиком, потому что `document_id`, `source_title` и
        `version` не задаются вызывающей стороной, а выводятся из реквизитов
        акта — разбирать его на части до сервиса значило бы вычислять их
        в нескольких местах.

        Args:
            request: Карточка документа: реквизиты акта, текст и метаданные.

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
        document_id = request.document_id
        category = request.category
        raw_text = request.raw_text
        document_metadata = DocumentMetadataInput(
            source_title=request.source_title,
            audience=request.audience,
            topics=request.topics,
            version=request.version,
            effective_date=request.content_date,
        )

        async with ingestion_stage('validation'):
            if len(raw_text) > MAX_RAW_TEXT_LENGTH:
                raise RawTextTooLargeError(document_id, len(raw_text), MAX_RAW_TEXT_LENGTH)
            if document_metadata.topics and category not in TOPICS_ALLOWED_CATEGORIES:
                raise TopicsNotAllowedForCategoryError(document_id, category, document_metadata.topics)

        async with self._document_lock(document_id):
            async with ingestion_stage('revision') as stage:
                await self._ensure_revision_is_not_stale(document_id, request.content_date)
                old_versions = await self._find_old_versions(document_id, document_metadata.version)
                stage.update(old_versions=old_versions)

            logger.info('🚀 Ingestion документа %s (version=%s).', document_id, document_metadata.version)
            embedded_chunks = await self._build_embedded_chunks(document_id, raw_text, category, document_metadata)

            expected_chunk_ids = self._embedded_chunk_ids(embedded_chunks)
            async with ingestion_stage('upsert') as stage:
                preexisting_chunk_ids = set(
                    await self.vector_store.get_document_version_chunk_ids(document_id, document_metadata.version)
                )
                stage.update(expected_chunks=len(expected_chunk_ids), preexisting_chunks=len(preexisting_chunk_ids))
                await self.vector_store.upsert_chunks(embedded_chunks, document_metadata)
            async with ingestion_stage('integrity') as stage:
                await self._verify_document_integrity(
                    document_id=document_id, version=document_metadata.version,
                    expected_chunk_ids=expected_chunk_ids, preexisting_chunk_ids=preexisting_chunk_ids,
                )
                stage.update(verified_chunks=len(expected_chunk_ids), identifiers_match=True)
            async with ingestion_stage('registry'):
                await self._save_document_record(request=request, document_metadata=document_metadata)
            async with ingestion_stage('retire') as stage:
                not_removed_versions = await self._delete_old_versions(document_id, old_versions)
                stage.update(old_versions=old_versions, not_removed_versions=not_removed_versions)
                if not_removed_versions:
                    stage.warning(f'Не удалены старые версии: {not_removed_versions}')
            async with ingestion_stage('monitoring'):
                await self._register_for_change_monitoring(request=request)

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

    async def _find_old_versions(self, document_id: str, current_version: str) -> list[str]:
        """Версии документа, уже проиндексированные в Qdrant под другим `version`."""
        return [
            version
            for version in await self.vector_store.get_document_versions(document_id)
            if version != current_version
        ]

    @asynccontextmanager
    async def _document_lock(self, document_id: str):
        async with AsyncExitStack() as stack:
            async with ingestion_stage('lock'):
                await stack.enter_async_context(self.document_repository.document_lock(document_id))
            yield

    async def _build_embedded_chunks(
        self, document_id: str, raw_text: str, category: Category, document_metadata: DocumentMetadataInput
    ) -> list[EmbeddedChunk]:
        """Препроцессинг → чанкинг → обогащение LLM → эмбеддинг (Этапы 1–4 плана).

        Raises:
            TooManyChunksError: Если документ дал больше `MAX_CHUNKS_PER_DOCUMENT` чанков.
        """
        async with ingestion_stage('preprocess') as stage:
            sections = preprocess_document(document_id, raw_text, category)
            stage.update(sections=len(sections))
        return await self._prepare_chunks(document_id, sections, document_metadata, 'document')

    async def _prepare_chunks(self, document_id, sections, document_metadata, scope):
        async with ingestion_stage('chunking') as stage:
            chunks = chunk_document(sections, version=document_metadata.version)
            self._ensure_unique_chunk_ids(document_id, document_metadata.version, scope, chunks)
            before_filter = len(chunks)
            chunks = [chunk for chunk in chunks if not is_editorial_note_only(chunk.text)]
            stage.update(chunks=len(chunks), filtered_chunks=before_filter - len(chunks))
            if not chunks:
                raise EmptyIngestionContentError(document_id)
            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                raise TooManyChunksError(document_id, len(chunks), MAX_CHUNKS_PER_DOCUMENT)
        async with ingestion_stage('enrichment', total=len(chunks), completed=0):
            enriched_chunks = await enrich_chunks(self.llm_client, chunks, document_metadata)
        async with ingestion_stage('embedding', total=len(enriched_chunks), completed=0):
            return await embed_chunks(self.embedding_client, enriched_chunks, get_settings().yandex.embedding_doc_model_uri)

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
        self, request: IngestRequest, document_metadata: DocumentMetadataInput
    ) -> None:
        """Пишет запись реестра документов (Этап 11.1). Источник правды о
        содержимом БЗ — Qdrant (upsert к этому моменту уже успешен), поэтому
        отказ записи сюда не должен ронять ingestion — перехватывается и
        логируется как предупреждение (FASTAPI_PATTERNS.md, раздел 9)."""
        try:
            await self.document_repository.save_document(
                Document(
                    document_id=request.document_id,
                    version=document_metadata.version,
                    category=request.category,
                    source_title=document_metadata.source_title,
                    audience=document_metadata.audience,
                    topics=document_metadata.topics,
                    act_type=request.act_type,
                    act_number=request.act_number,
                    act_date=request.act_date,
                    act_title=request.act_title,
                    act_authority=request.act_authority,
                    revision_date=request.revision_date,
                    is_active=True,
                )
            )
        except DocumentRepositoryError as error:
            stage_warning('registry', str(error))
            logger.warning(
                '⚠️ Не удалось записать реестр документа %s. Детали: %s', request.document_id, error,
            )

    async def ingest_section(
        self, document_id: str, section_number: str, request: SectionUpdateRequest,
    ) -> SectionUpdateResponse:
        trace = self._trace(request, document_id, section_number)
        async with trace.run():
            result = await self._ingest_section(document_id, section_number, request)
            trace.values['result'].update(result.model_dump(mode='json', exclude_unset=True))
        return result.model_copy(update=self._receipt(trace))

    async def _ingest_section(
        self,
        document_id: str,
        section_number: str,
        request: SectionUpdateRequest,
    ) -> SectionUpdateResponse:
        """Гранулярное обновление статьи с записью исхода в журнал изменений.

        Отказ фиксируется наравне с успехом и после этого пробрасывается
        дальше без изменений: вызывающая сторона (Legal Sync Service) должна
        увидеть ту же ошибку и решить, повторять ли попытку. Журнал нужен
        оператору — по нему видно, что сервис синхронизации присылал и чем
        это кончилось; журнал из одних удач не отличить от тишины.

        Args:
            document_id: Идентификатор документа.
            section_number: Номер статьи/пункта.
            request: Тело запроса с текстом, метаданными и версией.

        Returns:
            Сводка обновления.

        Raises:
            ValueError: Если category не поддерживает гранулярное обновление.
            TopicsNotAllowedForCategoryError: Если заданы недопустимые темы.
            LlmApiRequestError: Если обогащение чанка не удалось.
            EmbeddingApiRequestError: Если эмбеддинг чанка не удался.
        """
        try:
            async with self._document_lock(document_id):
                return await self._apply_section_update(document_id, section_number, request)
        except (ValueError, TopicsNotAllowedForCategoryError, StaleRevisionError) as error:
            # Данные отклонены проверками: тот же запрос будет отклонён снова,
            # повтор бессмыслен — событию нужен оператор.
            async with ingestion_stage('change_log', outcome=CHANGE_REJECTED):
                await self._write_change_log_entry(
                    document_id=document_id, section_number=section_number, request=request,
                    status=CHANGE_REJECTED, error=str(error),
                )
            raise
        except Exception as error:
            # Сорвалась обработка (LLM, эмбеддинги, векторное хранилище) —
            # повтор имеет смысл, поэтому исход отделён от отклонения.
            async with ingestion_stage('change_log', outcome=CHANGE_FAILED):
                await self._write_change_log_entry(
                    document_id=document_id, section_number=section_number, request=request,
                    status=CHANGE_FAILED, error=f'{type(error).__name__}: {error}',
                )
            raise

    async def _apply_section_update(
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
        async with ingestion_stage('validation'):
            if request.category not in SECTION_UPDATE_ALLOWED_CATEGORIES:
                raise ValueError(
                    f'Гранулярное обновление не поддерживается для category={request.category!r}. '
                    f'Допустимые: {sorted(SECTION_UPDATE_ALLOWED_CATEGORIES)}.'
                )
            if request.topics and request.category not in TOPICS_ALLOWED_CATEGORIES:
                raise TopicsNotAllowedForCategoryError(document_id, request.category, request.topics)

        parent_id = compute_parent_id(document_id, section_number)
        async with ingestion_stage('revision') as stage:
            await self._ensure_revision_is_not_stale(document_id, request.revision_date, parent_id)
            old_actual_chunk_ids = await self.vector_store.get_actual_section_chunk_ids(
                parent_id=parent_id, exclude_version=request.version,
            )
            stage.update(old_actual_chunks=len(old_actual_chunk_ids))

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
            amending_act=request.amending_act_reference,
            effective_date=request.revision_date,
        )

        embedded_chunks = await self._prepare_chunks(document_id, [section], document_metadata, f'parent_id={parent_id}')

        expected_chunk_ids = self._embedded_chunk_ids(embedded_chunks)
        async with ingestion_stage('upsert') as stage:
            preexisting_chunk_ids = set(await self.vector_store.get_section_version_chunk_ids(parent_id, request.version))
            stage.update(expected_chunks=len(expected_chunk_ids), preexisting_chunks=len(preexisting_chunk_ids))
            await self.vector_store.upsert_chunks(embedded_chunks, document_metadata)
        async with ingestion_stage('integrity') as stage:
            await self._verify_section_integrity(
                document_id=document_id, parent_id=parent_id, version=request.version,
                expected_chunk_ids=expected_chunk_ids, preexisting_chunk_ids=preexisting_chunk_ids,
            )
            stage.update(verified_chunks=len(expected_chunk_ids), identifiers_match=True)
        async with ingestion_stage('retire') as stage:
            superseded = await self.vector_store.set_chunks_inactive(
                chunk_ids=old_actual_chunk_ids, effective_until=request.revision_date,
            )
            stage.update(superseded_chunks=superseded)
        async with ingestion_stage('change_log'):
            await self._write_change_log_entry(
                document_id=document_id, section_number=section_number, request=request,
                chunks_count=len(embedded_chunks), superseded_chunks=superseded,
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

    # Блок интеграции с Legal Sync Service и журналом изменений

    async def _ensure_revision_is_not_stale(
        self, document_id: str, revision_date: date, parent_id: str | None = None,
    ) -> None:
        """Проверяется под блокировкой до обогащения и любых изменений Qdrant.

        Равная дата разрешена: повтор после потери ответа должен завершить
        тот же upsert и снятие старых чанков с актуальности.
        """
        current_revision_date = await self.vector_store.get_latest_revision_date(document_id, parent_id)
        if current_revision_date is not None and revision_date < current_revision_date:
            raise StaleRevisionError(document_id, revision_date, current_revision_date)

    async def _write_change_log_entry(
        self,
        document_id: str,
        section_number: str,
        request: SectionUpdateRequest,
        chunks_count: int = 0,
        superseded_chunks: int = 0,
        status: str = CHANGE_APPLIED,
        error: str | None = None,
    ) -> None:
        """Записывает исход попытки обновления статьи в журнал документа.

        Отказ самого журнала не меняет исход попытки: при успехе
        переиндексация уже выполнена в Qdrant, при отказе наверх уже летит
        исходная ошибка. Поэтому он логируется и не поднимается выше — иначе
        сбой журнала подменил бы собой настоящую причину отказа.
        """
        entry = DocumentChangeLog(
            document_id=document_id,
            section_number=section_number,
            section_title=request.section_title,
            category=request.category,
            version=request.version,
            revision_date=request.revision_date,
            amending_act_type=request.amending_act_type,
            amending_act_number=request.amending_act_number,
            amending_act_date=request.amending_act_date,
            chunks_count=chunks_count,
            superseded_chunks=superseded_chunks,
            status=status,
            error=error,
        )
        try:
            await self.change_log_repository.add_entry(entry)
        except DocumentChangeLogRepositoryError as log_error:
            stage_warning('change_log', str(log_error))
            logger.warning(
                '⚠️ Изменение %s/%s (%s) не записано в журнал: %s',
                document_id, section_number, status, log_error,
            )

    async def _register_for_change_monitoring(self, request: IngestRequest) -> None:
        """Ставит документ на контроль изменений в Legal Sync Service.

        Документ к этому моменту уже проиндексирован, поэтому недоступность
        сервиса мониторинга не отменяет ingestion. Но и молча пропустить это
        нельзя: без контроля база знаний будет отдавать устаревшую редакцию
        закона, не подавая никаких признаков проблемы, — поэтому отказ
        логируется как ошибка, требующая ручной постановки на контроль.
        """
        settings = get_settings().legal_sync
        trace = current_trace()
        stage = trace.steps['monitoring'] if trace else None
        if not settings.legal_sync_enabled:
            if stage:
                stage.skip('Мониторинг отключён настройкой.')
            return
        if request.category not in settings.tracked_categories:
            if stage:
                stage.skip('Категория не требует мониторинга.')
            logger.info(
                'ℹ️ Категория %s не отслеживается, документ %s на контроль не ставится.',
                request.category, request.document_id,
            )
            return
        if self.legal_sync_client is None:
            stage_warning('monitoring', 'Клиент Legal Sync не настроен.')
            logger.warning(
                '⚠️ Мониторинг включён, но клиент Legal Sync Service не собран. document_id=%s.',
                request.document_id,
            )
            return

        try:
            registration = await self.legal_sync_client.register_tracked_document(
                document_id=request.document_id,
                category=request.category,
                act_type=request.act_type,
                act_number=request.act_number,
                act_date=request.act_date,
                act_title=request.act_title,
                source_title=request.source_title,
                audience=request.audience,
                topics=request.topics,
                ingestion_id=trace.values['id'] if trace else None,
            )
            if stage and isinstance(registration, dict):
                stage.update(tracked_document_id=registration.get('id'), already_tracked=registration.get('already_tracked', False))
        except LegalSyncClientError as error:
            stage_warning('monitoring', str(error))
            logger.error(
                '❌ Документ %s не поставлен на контроль изменений, нужна ручная постановка: %s',
                request.document_id, error, exc_info=True,
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
        async with ingestion_stage('cleanup') as stage:
            stage.update(chunks=len(new_chunk_ids))
            try:
                await self.vector_store.delete_chunks(new_chunk_ids)
            except Exception as error:
                stage.warning(str(error))
                logger.exception('Не удалось удалить новые чанки после ошибки целостности.')

    async def _mark_old_versions_inactive(self, document_id: str, old_versions: list[str]) -> None:
        try:
            await self.document_repository.mark_versions_inactive(document_id, old_versions)
        except DocumentRepositoryError as error:
            stage_warning('retire', str(error))
            logger.warning(
                '⚠️ Не удалось обновить реестр документа %s после удаления старых версий. Детали: %s',
                document_id, error,
            )
