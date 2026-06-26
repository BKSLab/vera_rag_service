import json
from typing import Any, get_args
from urllib.parse import urlencode

from markupsafe import Markup, escape
from pydantic import ValidationError
from sqladmin import BaseView, ModelView, expose
from sqlalchemy import select
from starlette.requests import Request

from app.admin.csrf import get_or_create_csrf_token, verify_csrf_token
from app.admin.dashboard import get_dashboard_stats
from app.admin.services import build_documents_service, build_ingestion_service, build_search_service
from app.core.config_logger import logger
from app.db.models.document import Document
from app.db.models.search_log import SearchLog
from app.db.session import async_session_factory
from app.dependencies.vectorstore import get_vector_store
from app.exceptions.embedding import EmbeddingApiRequestError
from app.exceptions.ingestion import RawTextTooLargeError, TooManyChunksError
from app.exceptions.llm import LlmApiRequestError
from app.ingestion.extract import (
    MAX_UPLOAD_SIZE_BYTES,
    UnsupportedFileTypeError,
    UploadTooLargeError,
    extract_text_from_upload,
)
from app.models.metadata import CATEGORY_LABELS, Audience, Category
from app.models.schemas import DocumentMetadataInput, SearchFilters

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
    return Markup(f'<a href="/admin/document-chunks?{query}">{escape(model.document_id)}</a>')


class DocumentAdmin(ModelView, model=Document):
    """Реестр документов БЗ (Этап 11.1 плана) — список/история версий +
    удаление. Создание/редактирование — только через `DocumentUploadView`
    (запускает весь ingestion-пайплайн, не просто пишет строку в БД)."""

    name = 'Документ'
    name_plural = 'Документы в БЗ'
    icon = 'fa-solid fa-file-lines'

    column_list = [
        Document.id, Document.document_id, Document.version, Document.category,
        Document.source_title, Document.audience, Document.topic,
        Document.effective_date, Document.is_active, Document.created_at,
    ]
    column_searchable_list = [Document.document_id, Document.source_title]
    column_sortable_list = [Document.document_id, Document.effective_date, Document.created_at]
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


class DocumentUploadView(BaseView):
    """Загрузка документа в БЗ через /admin (Этап 11.1 плана) — закрывает
    разрыв: Expert/контент-менеджер не должен вручную собирать JSON с
    текстом документа внутри строки для `POST /ingest`."""

    name = 'Загрузка документа'
    icon = 'fa-solid fa-file-arrow-up'

    @expose('/document-upload', methods=['GET', 'POST'])
    async def document_upload(self, request: Request) -> Any:
        context: dict[str, Any] = {
            'categories': get_args(Category), 'audiences': get_args(Audience),
            'category_labels': CATEGORY_LABELS,
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
            document_metadata = DocumentMetadataInput(
                source_title=form.get('source_title', ''),
                audience=form.get('audience'),
                topic=form.get('topic', ''),
                version=form.get('version', ''),
                effective_date=form.get('effective_date'),
            )

            async with build_ingestion_service() as ingestion_service:
                result = await ingestion_service.ingest_document(
                    document_id=form.get('document_id', ''),
                    raw_text=raw_text,
                    category=category,
                    document_metadata=document_metadata,
                )
            context['success'] = (
                f"Документ «{result.document_id}» (версия {result.version}) проиндексирован: "
                f'{result.chunks_count} чанков. Замещено версий: {len(result.replaced_versions)}.'
            )
        except (ValueError, UnsupportedFileTypeError, ValidationError, RawTextTooLargeError, TooManyChunksError) as error:
            context['error'] = str(error)
        except (LlmApiRequestError, EmbeddingApiRequestError) as error:
            context['error'] = str(error)

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
        return await self.templates.TemplateResponse(request, 'dashboard.html', {'stats': stats})
