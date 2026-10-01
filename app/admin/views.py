import json
from typing import Any, get_args
from urllib.parse import urlencode

from markupsafe import Markup, escape
from pydantic import ValidationError
from sqladmin import BaseView, ModelView, expose
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from starlette.requests import Request

from app.admin.csrf import get_or_create_csrf_token, verify_csrf_token
from app.admin.dashboard import get_dashboard_stats
from app.admin.services import build_documents_service, build_ingestion_service, build_search_service
from app.core.config_logger import logger
from app.db.models.document import Document
from app.db.models.document_change_log import (
    CHANGE_APPLIED,
    CHANGE_FAILED,
    CHANGE_REJECTED,
    DocumentChangeLog,
)
from app.db.models.ingestion_run import IngestionRun
from app.db.models.search_log import SearchLog
from app.db.models.topic import Topic
from app.db.session import async_session_factory
from app.dependencies.vectorstore import get_vector_store
from app.exceptions.embedding import EmbeddingApiRequestError
from app.exceptions.ingestion import (
    RawTextTooLargeError,
    StaleRevisionError,
    TooManyChunksError,
    TopicsNotAllowedForCategoryError,
)
from app.exceptions.llm import LlmApiRequestError
from app.ingestion.extract import (
    MAX_UPLOAD_SIZE_BYTES,
    UnsupportedFileTypeError,
    UploadTooLargeError,
    extract_text_from_upload,
)
from app.models.metadata import CATEGORY_LABELS, Audience, Category
from app.models.requisites import (
    ACT_TYPES_BY_CATEGORY,
    requisites_required,
    revision_applicable,
)
from app.models.schemas import (
    REQUIRED_NUMBER_SUFFIX_BY_CATEGORY,
    TOPICS_ALLOWED_CATEGORIES,
    IngestRequest,
    SearchFilters,
)
from app.services.ingestion_history import run_view
from app.services.ingestion_journal import STATUS_LABELS

SEARCH_TEST_TOP_K = 5

_AUDIENCE_COLORS = {'seeker': '#3B82F6', 'employer': '#F5B800', 'both': '#22C55E'}

_JSON_STYLE = 'white-space:pre-wrap;word-break:break-word;max-width:900px;display:block;'


def _fmt_audience(model: SearchLog, attr: str) -> Markup:
    if not model.audience:
        return Markup('<em>—</em>')
    color = _AUDIENCE_COLORS.get(model.audience, '#888')
    return Markup(
        f'<span style="background:{color};color:#000;padding:2px 8px;'
        f'border-radius:4px;font-size:0.8em;font-weight:600;">{model.audience}</span>'
    )


def _fmt_json(model: SearchLog, attr: str) -> Markup:
    """`json.dumps` экранирует только то, что нужно для валидности самой JSON-строки —
    не HTML-спецсимволы. Значение может содержать текст реальных документов/LLM-вывод
    (`final_response` и т.п.), поэтому перед вставкой в `<pre>` экранируем явно
    (см. AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md, ADM-1/SEC-3 — stored XSS)."""
    value = getattr(model, attr, None)
    pretty = json.dumps(value, ensure_ascii=False, indent=2)
    return Markup(f"<pre style='{_JSON_STYLE}'>{escape(pretty)}</pre>")


class SearchLogAdmin(ModelView, model=SearchLog):
    """Журнал поисковых запросов `/search` (Этап 8 плана) — для анализа
    качества поиска: вопрос → кандидаты на каждой стадии → финальный ответ."""

    name = 'Поисковый запрос'
    name_plural = 'Журнал поисковых запросов'
    icon = 'fa-solid fa-magnifying-glass'

    column_list = [
        SearchLog.id,
        SearchLog.query,
        SearchLog.audience,
        SearchLog.topic,
        SearchLog.category,
        SearchLog.latency_query_expansion_ms,
        SearchLog.latency_embed_query_ms,
        SearchLog.latency_hybrid_search_ms,
        SearchLog.latency_rerank_ms,
        SearchLog.created_at,
    ]
    column_searchable_list = [SearchLog.query, SearchLog.request_id, SearchLog.topic]
    column_sortable_list = [
        SearchLog.created_at,
        SearchLog.latency_query_expansion_ms,
        SearchLog.latency_embed_query_ms,
        SearchLog.latency_hybrid_search_ms,
        SearchLog.latency_rerank_ms,
    ]
    column_default_sort = [(SearchLog.created_at, True)]

    column_formatters = {SearchLog.audience: _fmt_audience}
    column_formatters_detail = {
        SearchLog.audience: _fmt_audience,
        SearchLog.query_variants: _fmt_json,
        SearchLog.dense_candidates: _fmt_json,
        SearchLog.sparse_candidates: _fmt_json,
        SearchLog.rrf_candidates: _fmt_json,
        SearchLog.reranked_chunk_ids: _fmt_json,
        SearchLog.final_response: _fmt_json,
    }

    can_create = False
    can_edit = False
    can_delete = True


def _document_id_link(model: Document) -> Markup:
    """Превращает `document_id` в ссылку на просмотр чанков этой версии в
    Qdrant (`DocumentChunksView`) — без этого нет способа увидеть реальный
    проиндексированный текст, а не только метаданные реестра."""
    query = urlencode({'document_id': model.document_id, 'version': model.version})
    history = urlencode({'document_id': model.document_id})
    return Markup(f'<a href="/admin/document-chunks?{escape(query)}">{escape(model.document_id)}</a>'
                  f'<div><a href="/admin/ingestion-log?{escape(history)}">История обновлений</a></div>')


def build_ingest_request(form: Any, category: str, raw_text: str) -> IngestRequest:
    """Собирает `IngestRequest` из полей формы карточки загрузки.

    Форма отдаёт только то, что показано оператору для выбранной категории:
    у авторских материалов нет вида акта и номера, у судебной практики нет
    действующей редакции. Недостающие поля приходят пустыми, а не выдуманными.

    Args:
        form: Разобранное тело формы.
        category: Проверенная категория источника.
        raw_text: Текст, извлечённый из загруженного файла.

    Returns:
        Провалидированный запрос на индексацию.

    Raises:
        ValidationError: Реквизиты не соответствуют правилам категории.
    """
    return IngestRequest(
        category=category,
        act_type=_optional_form_value(form, 'act_type'),
        act_number=_optional_form_value(form, 'act_number'),
        act_date=form.get('act_date'),
        act_title=form.get('act_title', ''),
        act_authority=_optional_form_value(form, 'act_authority'),
        source_title=form.get('source_title', ''),
        revision_date=_optional_form_value(form, 'revision_date'),
        document_id=_optional_form_value(form, 'document_id'),
        raw_text=raw_text,
        audience=form.get('audience'),
        topics=form.getlist('topics'),
    )


def _change_status_badge(model: DocumentChangeLog) -> Markup:
    """Красит исход попытки: неприменённое изменение должно быть видно
    при беглом просмотре списка, а не вычитываться из колонки текстом."""
    colors = {CHANGE_APPLIED: 'green', CHANGE_REJECTED: 'orange', CHANGE_FAILED: 'red'}
    color = colors.get(model.status, 'secondary')
    return Markup(f'<span class="badge bg-{color}-lt">{escape(model.status)}</span>')


def _optional_form_value(form: Any, name: str) -> str | None:
    """Возвращает значение поля формы или `None`, если оно пустое.

    Пустая строка из HTML-формы и «значение не задано» — разные вещи для
    Pydantic: без этого пустое необязательное поле упало бы на валидации типа.
    """
    value = form.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


class DocumentChangeLogAdmin(ModelView, model=DocumentChangeLog):
    """Журнал изменений документов БЗ — что и когда переиндексировано.

    Только чтение: журнал фиксирует состоявшиеся факты, редактировать его
    задним числом нельзя. Записи появляются при каждой попытке гранулярного
    обновления статьи из Legal Sync Service — и применённой, и отклонённой.

    Это единственное место, где видно, что сервис синхронизации присылал:
    сам он живёт отдельно, и без журнала неприменённое изменение выглядело бы
    так же, как отсутствие изменений.
    """

    name = 'Изменение'
    name_plural = 'Журнал изменений'
    icon = 'fa-solid fa-clock-rotate-left'

    column_list = [
        DocumentChangeLog.id, DocumentChangeLog.created_at, DocumentChangeLog.status,
        DocumentChangeLog.document_id, DocumentChangeLog.section_number,
        DocumentChangeLog.section_title, DocumentChangeLog.category, DocumentChangeLog.version,
        DocumentChangeLog.revision_date, DocumentChangeLog.amending_act_type,
        DocumentChangeLog.amending_act_number, DocumentChangeLog.amending_act_date,
        DocumentChangeLog.chunks_count, DocumentChangeLog.superseded_chunks,
        DocumentChangeLog.error,
    ]
    # Статус в поиске — чтобы отобрать неприменённые изменения запросом
    # `rejected`/`failed`: именно они требуют реакции оператора.
    column_searchable_list = [
        DocumentChangeLog.document_id, DocumentChangeLog.section_number, DocumentChangeLog.status,
    ]
    column_sortable_list = [
        DocumentChangeLog.status,
        DocumentChangeLog.document_id,
        DocumentChangeLog.revision_date,
        DocumentChangeLog.created_at,
    ]
    column_default_sort = [(DocumentChangeLog.created_at, True)]
    column_formatters = {
        DocumentChangeLog.status: lambda model, attr: _change_status_badge(model),
        DocumentChangeLog.document_id: lambda model, attr: _document_id_link(model),
    }
    column_formatters_detail = {
        DocumentChangeLog.status: lambda model, attr: _change_status_badge(model),
        DocumentChangeLog.document_id: lambda model, attr: _document_id_link(model),
    }

    can_create = False
    can_edit = False
    can_delete = False


class DocumentAdmin(ModelView, model=Document):
    """Реестр документов БЗ (Этап 11.1 плана) — список/история версий +
    удаление. Создание/редактирование — только через `DocumentUploadView`
    (запускает весь ingestion-пайплайн, не просто пишет строку в БД)."""

    name = 'Документ'
    name_plural = 'Документы в БЗ'
    icon = 'fa-solid fa-file-lines'

    column_list = [
        Document.id, Document.document_id, Document.category, Document.act_type,
        Document.act_number, Document.act_date, Document.act_title,
        Document.act_authority, Document.revision_date, Document.version,
        Document.source_title, Document.audience, Document.topics,
        Document.is_active, Document.created_at,
    ]
    column_searchable_list = [Document.document_id, Document.source_title]
    column_sortable_list = [Document.document_id, Document.act_date, Document.revision_date, Document.created_at]
    column_default_sort = [(Document.created_at, True)]
    column_formatters = {Document.document_id: lambda model, attr: _document_id_link(model)}
    column_formatters_detail = {Document.document_id: lambda model, attr: _document_id_link(model)}

    can_create = False
    can_edit = False
    can_delete = True

    async def delete_model(self, request: Request, pk: Any) -> None:
        """Удаление одной строки реестра — это удаление этой версии документа
        из БЗ целиком, не просто строки: убирает и чанки из Qdrant (источник
        правды о содержимом БЗ), и саму запись (раздел 11.1 плана).

        Через `DocumentsService.delete_document` — тот же код, что и у
        публичного `DELETE /document/{id}` (ARCH-4,
        AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md), не дублирующая
        логика через `super().delete_model()` — иначе два пути удаления
        документа расходятся в том, что каждый из них реально удаляет.
        """
        document = await self.get_object_for_delete(pk)
        if document is None:
            return
        async with build_documents_service() as service:
            await service.delete_document(document.document_id, version=document.version)
        # ADM-2 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — единая
        # учётная запись админки не различает личность, но IP+момент
        # действия уже сокращают время расследования инцидента (например,
        # дублирование из-за двух одновременных загрузок, ING-2).
        logger.info(
            '🗑️ [admin] Удаление документа %s (версия %s) через админку. IP: %s.',
            document.document_id, document.version, request.client.host if request.client else '-',
        )


class TopicAdmin(ModelView, model=Topic):
    """Справочник тем документов (раздел 3 плана) — управляется полностью
    через админку, без деплоя кода (в отличие от `Category`/`Audience`,
    захардкоженных в `app/models/metadata.py`). Осмысленны только для
    `other_npa`/`case_law`/`authorial` — см. `TOPICS_ALLOWED_CATEGORIES`,
    `TopicsNotAllowedForCategoryError`."""

    name = 'Тема'
    name_plural = 'Темы документов'
    icon = 'fa-solid fa-tags'

    column_list = [Topic.id, Topic.name, Topic.comment, Topic.created_at]
    column_searchable_list = [Topic.name]
    column_sortable_list = [Topic.name, Topic.created_at]
    form_columns = [Topic.name, Topic.comment]

    can_create = True
    can_edit = True
    can_delete = True


class DocumentUploadView(BaseView):
    """Загрузка документа в БЗ через /admin (Этап 11.1 плана) — закрывает
    разрыв: Expert/контент-менеджер не должен вручную собирать JSON с
    текстом документа внутри строки для `POST /ingest`."""

    name = 'Загрузка документа'
    icon = 'fa-solid fa-file-arrow-up'

    @expose('/document-upload', methods=['GET', 'POST'])
    async def document_upload(self, request: Request) -> Any:
        async with async_session_factory() as db_session:
            result = await db_session.execute(select(Topic.name).order_by(Topic.name))
            topic_names = [row[0] for row in result.all()]

        context: dict[str, Any] = {
            'categories': get_args(Category), 'audiences': get_args(Audience),
            'category_labels': CATEGORY_LABELS,
            'topic_names': topic_names,
            'topics_allowed_categories': sorted(TOPICS_ALLOWED_CATEGORIES),
            'act_types_by_category': {
                category: list(types) for category, types in ACT_TYPES_BY_CATEGORY.items()
            },
            'requisites_categories': [
                category for category in get_args(Category) if requisites_required(category)
            ],
            'revision_categories': [
                category for category in get_args(Category) if revision_applicable(category)
            ],
            'required_number_suffix': REQUIRED_NUMBER_SUFFIX_BY_CATEGORY,
            'csrf_token': get_or_create_csrf_token(request),
        }

        if request.method == 'GET':
            return await self.templates.TemplateResponse(request, 'document_upload.html', context)

        try:
            # Проверка `Content-Length` до парсинга формы (ADM-6/ING-7/SEC-4)
            # — `request.form()` сам буферизует всё тело запроса в память;
            # без этой проверки `extract_text_from_upload`'s проверка размера
            # сработала бы только после того, как тело уже целиком прочитано.
            content_length = request.headers.get('content-length')
            if content_length is not None and int(content_length) > MAX_UPLOAD_SIZE_BYTES:
                raise UploadTooLargeError(int(content_length), MAX_UPLOAD_SIZE_BYTES)

            form = await request.form()
            if not verify_csrf_token(request, form.get('csrf_token')):
                raise ValueError('Невалидный CSRF-токен — обновите страницу и попробуйте снова.')

            upload = form.get('file')
            if upload is None or not getattr(upload, 'filename', None):
                raise ValueError('Файл не выбран.')

            category = form.get('category')
            if category not in get_args(Category):
                raise ValueError(f'Недопустимая категория: {category!r}.')

            raw_text = extract_text_from_upload(upload.filename, await upload.read())
            ingest_request = build_ingest_request(form, category, raw_text)
            context['history_url'] = '/admin/ingestion-log?' + urlencode({'document_id': ingest_request.document_id})

            async with build_ingestion_service() as ingestion_service:
                result = await ingestion_service.ingest_document(request=ingest_request)
            context['operation_id'] = result.operation_id
            context['warnings'] = result.warnings
            context['success'] = (
                f"Документ «{result.document_id}» (версия {result.version}) проиндексирован: "
                f'{result.chunks_count} чанков. Замещено версий: {len(result.replaced_versions)}.'
            )
        except (
            ValueError, UnsupportedFileTypeError, ValidationError,
            RawTextTooLargeError, TooManyChunksError, TopicsNotAllowedForCategoryError, StaleRevisionError,
        ) as error:
            context['error'] = str(error)
        except (LlmApiRequestError, EmbeddingApiRequestError) as error:
            context['error'] = str(error)
        except Exception:
            logger.exception('Ошибка загрузки документа через админку.')
            context['error'] = 'Обработка завершилась ошибкой. Откройте историю обновлений для проверки этапов.'

        return await self.templates.TemplateResponse(request, 'document_upload.html', context)


class DocumentChunksView(BaseView):
    """Просмотр реально проиндексированных чанков документа в Qdrant —
    `DocumentAdmin` показывает только метаданные реестра в Postgres, не
    сам текст/синтетический заголовок/гипотетические вопросы чанков."""

    name = 'Чанки документа'
    icon = 'fa-solid fa-list-ul'

    @expose('/document-chunks', methods=['GET'])
    async def document_chunks(self, request: Request) -> Any:
        document_id = (request.query_params.get('document_id') or '').strip()
        version = (request.query_params.get('version') or '').strip() or None

        async with async_session_factory() as db_session:
            result = await db_session.execute(select(Document.document_id).distinct().order_by(Document.document_id))
            document_ids = [row[0] for row in result.all()]

        context: dict[str, Any] = {'document_id': document_id, 'version': version, 'document_ids': document_ids}

        if document_id:
            context['chunks'] = await get_vector_store().list_chunks(document_id, version=version)

        return await self.templates.TemplateResponse(request, 'document_chunks.html', context)


class SearchTestView(BaseView):
    """Интерактивное тестирование поиска через /admin (Этап 11.2 плана) —
    позволяет задать вопрос и увидеть кандидатов на каждой стадии
    (dense/sparse/RRF/rerank) без поднятия Agent Service/MCP Tools Server.
    Вызывает `SearchService.search_with_diagnostics` напрямую — тот же
    сервис, что и `POST /search`, без дублирования логики поиска; каждый
    тестовый прогон автоматически попадает в `search_logs` (Этап 8)."""

    name = 'Тестирование поиска'
    icon = 'fa-solid fa-flask'

    @expose('/search-test', methods=['GET', 'POST'])
    async def search_test(self, request: Request) -> Any:
        context: dict[str, Any] = {
            'categories': get_args(Category), 'audiences': get_args(Audience),
            'category_labels': CATEGORY_LABELS,
            'csrf_token': get_or_create_csrf_token(request),
        }

        if request.method == 'GET':
            return await self.templates.TemplateResponse(request, 'search_test.html', context)

        form = await request.form()
        query = (form.get('query') or '').strip()
        context.update(
            query=query,
            selected_audience=form.get('audience') or '',
            selected_topic=form.get('topic') or '',
            selected_category=form.get('category') or '',
        )

        try:
            if not verify_csrf_token(request, form.get('csrf_token')):
                raise ValueError('Невалидный CSRF-токен — обновите страницу и попробуйте снова.')
            if not query:
                raise ValueError('Введите текст запроса.')

            filters = SearchFilters(
                audience=form.get('audience') or None,
                topic=form.get('topic') or None,
                category=form.get('category') or None,
            )
            async with build_search_service() as search_service:
                context['diagnostics'] = await search_service.search_with_diagnostics(
                    query=query, filters=filters, top_k=SEARCH_TEST_TOP_K
                )
        except (ValueError, ValidationError) as error:
            context['error'] = str(error)
        except (EmbeddingApiRequestError, LlmApiRequestError) as error:
            context['error'] = str(error)

        return await self.templates.TemplateResponse(request, 'search_test.html', context)


class DashboardView(BaseView):
    """Сводный мониторинг сервиса через /admin — без неё единственный
    способ оценить состояние БЗ и поиска — листать сырые списки построчно
    в `DocumentAdmin`/`SearchLogAdmin` (расширение Этапа 11 плана)."""

    name = 'Дашборд'
    icon = 'fa-solid fa-gauge-high'

    @expose('/dashboard', methods=['GET'])
    async def dashboard(self, request: Request) -> Any:
        async with async_session_factory() as db_session:
            stats = await get_dashboard_stats(db_session, get_vector_store())
        recent_runs, ingestion_log_error = [], None
        if stats.postgres_ok:
            try:
                async with async_session_factory() as db_session:
                    recent_runs = [run_view(row) for row in (await db_session.scalars(
                        select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(5)
                    )).all()]
            except (SQLAlchemyError, OSError):
                ingestion_log_error = 'Журнал обновлений временно недоступен.'
        return await self.templates.TemplateResponse(request, 'dashboard.html', {
            'stats': stats, 'recent_runs': recent_runs, 'run_status_labels': STATUS_LABELS,
            'ingestion_log_error': ingestion_log_error,
        })
