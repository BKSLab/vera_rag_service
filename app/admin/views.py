import json

from markupsafe import Markup
from sqladmin import ModelView

from app.db.models.search_log import SearchLog

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
