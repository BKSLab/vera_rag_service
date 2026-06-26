from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class SearchLog(Base):
    """Журнал поисковых запросов `POST /search` (Этап 8 плана).

    Журнал событий — записи неизменяемы после создания, поэтому нет
    `updated_at` (см. FASTAPI_PATTERNS.md, раздел 11). Хранит вход запроса,
    промежуточные кандидаты на каждой стадии (dense/sparse/RRF/rerank) и
    финальный ответ — для оценки релевантности и отладки поиска (раздел 3
    плана, WBS 5.4/5.6), не для аудита диалога (это зона Agent Service).
    """

    __tablename__ = 'search_logs'

    id: Mapped[int] = mapped_column(primary_key=True, comment='Уникальный идентификатор записи журнала.')
    request_id: Mapped[str] = mapped_column(
        String(length=36), nullable=False, comment='UUID конкретного запроса — для сопоставления со структурными логами приложения.'
    )
    query: Mapped[str] = mapped_column(Text, nullable=False, comment='Исходный текст поискового запроса.')
    query_variants: Mapped[list] = mapped_column(
        JSONB, nullable=False,
        comment='Подвопросы/переформулировки после расширения запроса (раздел 8 плана): список текстов, '
        'каждый прошёл собственный hybrid_search до фьюжна в rrf_candidates.',
    )
    audience: Mapped[str | None] = mapped_column(String(length=20), nullable=True, comment='Значение фильтра audience, если был задан.')
    topic: Mapped[str | None] = mapped_column(String(length=100), nullable=True, comment='Значение фильтра topic, если был задан.')
    category: Mapped[str | None] = mapped_column(String(length=20), nullable=True, comment='Значение фильтра category, если был задан.')
    dense_candidates: Mapped[list] = mapped_column(JSONB, nullable=False, comment='Top-20 кандидатов dense-поиска до фьюжна: [[chunk_id, score], ...].')
    sparse_candidates: Mapped[list] = mapped_column(JSONB, nullable=False, comment='Top-20 кандидатов sparse/BM25-поиска до фьюжна: [[chunk_id, score], ...].')
    rrf_candidates: Mapped[list] = mapped_column(JSONB, nullable=False, comment='Результат RRF fusion: [[chunk_id, score], ...].')
    reranked_chunk_ids: Mapped[list] = mapped_column(JSONB, nullable=False, comment='chunk_id в порядке, который вернул LLM-reranker.')
    final_response: Mapped[list] = mapped_column(JSONB, nullable=False, comment='Финальный список чанков, отданный клиенту /search.')
    latency_query_expansion_ms: Mapped[float] = mapped_column(
        Float, nullable=False, comment='Латентность стадии расширения запроса (декомпозиция+переформулировка, раздел 8 плана), мс.'
    )
    latency_embed_query_ms: Mapped[float] = mapped_column(Float, nullable=False, comment='Латентность стадии embed_query (сумма по всем вариантам запроса), мс.')
    latency_hybrid_search_ms: Mapped[float] = mapped_column(Float, nullable=False, comment='Латентность стадии hybrid_search по всем вариантам запроса (dense+sparse+RRF на вариант + слияние), мс.')
    latency_rerank_ms: Mapped[float] = mapped_column(Float, nullable=False, comment='Латентность стадии rerank, мс.')
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment='Момент выполнения поискового запроса.'
    )

    def __repr__(self) -> str:
        return f"<SearchLog(id={self.id}, request_id='{self.request_id}', query='{self.query[:30]}')>"
