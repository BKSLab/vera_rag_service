import json
from typing import Any, get_args

from markupsafe import Markup
from pydantic import ValidationError
from sqladmin import BaseView, ModelView, expose
from starlette.requests import Request

from app.admin.services import build_ingestion_service, build_search_service
from app.db.models.document import Document
from app.db.models.search_log import SearchLog
from app.dependencies.vectorstore import get_vector_store
from app.exceptions.embedding import EmbeddingApiRequestError
from app.exceptions.llm import LlmApiRequestError
from app.ingestion.extract import UnsupportedFileTypeError, extract_text_from_upload
from app.models.metadata import Audience, Category
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
    value = getattr(model, attr, None)
    pretty = json.dumps(value, ensure_ascii=False, indent=2)
    return Markup(f"<pre style='{_JSON_STYLE}'>{pretty}</pre>")


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
        SearchLog.latency_embed_query_ms,
        SearchLog.latency_hybrid_search_ms,
        SearchLog.latency_rerank_ms,
        SearchLog.created_at,
    ]
    column_searchable_list = [SearchLog.query, SearchLog.request_id, SearchLog.topic]
    column_sortable_list = [
        SearchLog.created_at,
        SearchLog.latency_embed_query_ms,
        SearchLog.latency_hybrid_search_ms,
        SearchLog.latency_rerank_ms,
    ]
    column_default_sort = [(SearchLog.created_at, True)]

    column_formatters = {SearchLog.audience: _fmt_audience}
    column_formatters_detail = {
        SearchLog.audience: _fmt_audience,
        SearchLog.dense_candidates: _fmt_json,
        SearchLog.sparse_candidates: _fmt_json,
        SearchLog.rrf_candidates: _fmt_json,
        SearchLog.reranked_chunk_ids: _fmt_json,
        SearchLog.final_response: _fmt_json,
    }

    can_create = False
    can_edit = False
    can_delete = True


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

    can_create = False
    can_edit = False
    can_delete = True

    async def delete_model(self, request: Request, pk: Any) -> None:
        """Удаление в админке — это удаление документа из БЗ целиком, не
        просто строки реестра: убирает и чанки из Qdrant (источник правды
        о содержимом БЗ), и саму запись (раздел 11.1 плана)."""
        document = await self.get_object_for_delete(pk)
        if document is not None:
            await get_vector_store().delete_document(document.document_id, version=document.version)
        await super().delete_model(request, pk)


class DocumentUploadView(BaseView):
    """Загрузка документа в БЗ через /admin (Этап 11.1 плана) — закрывает
    разрыв: Expert/контент-менеджер не должен вручную собирать JSON с
    текстом документа внутри строки для `POST /ingest`."""

    name = 'Загрузка документа'
    icon = 'fa-solid fa-file-arrow-up'

    @expose('/document-upload', methods=['GET', 'POST'])
    async def document_upload(self, request: Request) -> Any:
        context: dict[str, Any] = {'categories': get_args(Category), 'audiences': get_args(Audience)}

        if request.method == 'GET':
            return await self.templates.TemplateResponse(request, 'document_upload.html', context)

        form = await request.form()
        upload = form.get('file')

        try:
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
        except (ValueError, UnsupportedFileTypeError, ValidationError) as error:
            context['error'] = str(error)
        except (LlmApiRequestError, EmbeddingApiRequestError) as error:
            context['error'] = str(error)

        return await self.templates.TemplateResponse(request, 'document_upload.html', context)


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
        context: dict[str, Any] = {'categories': get_args(Category), 'audiences': get_args(Audience)}

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
