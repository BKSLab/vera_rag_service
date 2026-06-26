import asyncio
import time
from dataclasses import dataclass
from uuid import uuid4

from app.clients.embeddings import EmbeddingClient
from app.clients.llm import LlmClient
from app.core.config_logger import logger
from app.core.request_context import get_request_id
from app.core.settings import get_settings
from app.db.models.search_log import SearchLog
from app.exceptions.search_log import SearchLogRepositoryError
from app.models.schemas import SearchFilters, SearchResultChunk
from app.repositories.search_log import SearchLogRepository
from app.search.fusion import rrf_fusion
from app.search.hybrid import HybridSearchResult, hybrid_search
from app.search.query_expansion import expand_query
from app.search.reranker import rerank_chunks
from app.vectorstore.qdrant_client import (
    SYNTHETIC_TITLE_PAYLOAD_FIELD,
    TEXT_PAYLOAD_FIELD,
    QdrantVectorStore,
)


@dataclass
class SearchDiagnostics:
    """Полный результат поиска по стадиям (Этап 11.2 плана) — для
    интерактивного тестирования в админке: dense/sparse top-K до фьюжна,
    RRF-склейка, порядок reranker'а и финальный список. `search()` —
    тонкая обёртка, возвращающая только `results`, чтобы не дублировать
    логику между `/search` и страницей тестирования в админке."""

    dense: list[tuple[str, float]]
    sparse: list[tuple[str, float]]
    fused: list[tuple[str, float]]
    reranked_ids: list[str]
    results: list[SearchResultChunk]
    query_variants: list[str]


class SearchService:
    """Оркестратор поискового запроса: расширение запроса → embed_query →
    hybrid search → rerank (раздел 8 плана, Этапы 4–6, 8)."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        reranker_llm_client: LlmClient,
        query_expansion_llm_client: LlmClient,
        vector_store: QdrantVectorStore,
        search_log_repository: SearchLogRepository,
    ):
        self.embedding_client = embedding_client
        self.reranker_llm_client = reranker_llm_client
        self.query_expansion_llm_client = query_expansion_llm_client
        self.vector_store = vector_store
        self.search_log_repository = search_log_repository

    async def search(
        self, query: str, filters: SearchFilters, top_k: int
    ) -> list[SearchResultChunk]:
        """Выполняет полный поисковый запрос и возвращает финальные результаты.

        Args:
            query: Текст запроса пользователя.
            filters: Фильтры по метаданным.
            top_k: Сколько чанков вернуть после переранжирования.

        Returns:
            Чанки, отсортированные по релевантности. Пустой список, если
            ничего не нашлось после фильтрации по метаданным.

        Raises:
            EmbeddingApiRequestError: Если эмбеддинг запроса не удался.
        """
        diagnostics = await self.search_with_diagnostics(query, filters, top_k)
        return diagnostics.results

    async def search_with_diagnostics(
        self, query: str, filters: SearchFilters, top_k: int
    ) -> SearchDiagnostics:
        """Как `search`, но возвращает промежуточные данные всех стадий
        (Этап 11.2 плана) — нужно странице интерактивного тестирования
        поиска в админке, которая показывает dense/sparse/RRF/rerank, не
        только финальный список.
        """
        # LOG-2 — request_id из контекста HTTP-запроса (middleware,
        # app/main.py), не генерируется заново здесь — иначе строка лога
        # на уровне эндпоинта и запись в search_logs для одного и того же
        # запроса получали бы два разных идентификатора. Fallback на новый
        # uuid4, если сервис вызван не через HTTP (юнит-тесты, скрипты).
        request_id = get_request_id() or str(uuid4())

        started_at = time.perf_counter()
        query_variants = await expand_query(self.query_expansion_llm_client, query)
        latency_query_expansion_ms = (time.perf_counter() - started_at) * 1000

        # Каждый вариант запроса (исходный/декомпозированный подвопрос ×
        # переформулировка, раздел 8 плана) получает собственный
        # embed_query + hybrid_search — оба запускаются параллельно по
        # вариантам (`MAX_SUB_QUESTIONS` × `MAX_REPHRASINGS_PER_SUB_QUESTION`
        # ограничивает веер, см. app/models/schemas.py), стадии остаются
        # раздельно измеримыми, как и до расширения запроса.
        started_at = time.perf_counter()
        query_vectors = await asyncio.gather(
            *(
                self.embedding_client.get_embedding(
                    text=variant, model_uri=get_settings().yandex.embedding_query_model_uri
                )
                for variant in query_variants
            )
        )
        latency_embed_query_ms = (time.perf_counter() - started_at) * 1000

        started_at = time.perf_counter()
        variant_hybrid_results = await asyncio.gather(
            *(
                hybrid_search(
                    self.vector_store.client, self.vector_store.collection_name, vector, variant, filters
                )
                for vector, variant in zip(query_vectors, query_variants, strict=True)
            )
        )
        latency_hybrid_search_ms = (time.perf_counter() - started_at) * 1000

        dense = [item for result in variant_hybrid_results for item in result.dense]
        sparse = [item for result in variant_hybrid_results for item in result.sparse]
        # Слияние вариантов запроса через ту же RRF, что уже фьюзит
        # категорийные ленты внутри одного hybrid_search (Этап 5.1) —
        # каждый вариант — ещё один ранжированный список chunk_id.
        fused = rrf_fusion([[chunk_id for chunk_id, _ in result.fused] for result in variant_hybrid_results])
        hybrid_result = HybridSearchResult(dense=dense, sparse=sparse, fused=fused)

        if not hybrid_result.fused:
            await self._save_search_log(
                request_id, query, query_variants, filters, hybrid_result, reranked_ids=[], results=[],
                latency_query_expansion_ms=latency_query_expansion_ms,
                latency_embed_query_ms=latency_embed_query_ms,
                latency_hybrid_search_ms=latency_hybrid_search_ms,
                latency_rerank_ms=0.0,
            )
            return SearchDiagnostics(
                dense=hybrid_result.dense, sparse=hybrid_result.sparse, fused=hybrid_result.fused,
                reranked_ids=[], results=[], query_variants=query_variants,
            )

        rrf_scores = dict(hybrid_result.fused)
        candidate_ids = [chunk_id for chunk_id, _ in hybrid_result.fused]

        points = await self.vector_store.client.retrieve(
            collection_name=self.vector_store.collection_name, ids=candidate_ids, with_payload=True
        )
        payload_by_id = {str(point.id): point.payload for point in points}

        candidates_for_rerank = [
            (chunk_id, payload_by_id[chunk_id][TEXT_PAYLOAD_FIELD])
            for chunk_id in candidate_ids
            if chunk_id in payload_by_id
        ]
        started_at = time.perf_counter()
        reranked_ids = await rerank_chunks(self.reranker_llm_client, query, candidates_for_rerank, top_n=top_k)
        latency_rerank_ms = (time.perf_counter() - started_at) * 1000

        results = [
            SearchResultChunk(
                chunk_id=chunk_id,
                text=payload_by_id[chunk_id][TEXT_PAYLOAD_FIELD],
                synthetic_title=payload_by_id[chunk_id][SYNTHETIC_TITLE_PAYLOAD_FIELD],
                source_title=payload_by_id[chunk_id]['source_title'],
                audience=payload_by_id[chunk_id]['audience'],
                topic=payload_by_id[chunk_id]['topic'],
                category=payload_by_id[chunk_id]['category'],
                section_number=payload_by_id[chunk_id].get('section_number'),
                section_title=payload_by_id[chunk_id].get('section_title'),
                score=rrf_scores.get(chunk_id, 0.0),
            )
            for chunk_id in reranked_ids
            if chunk_id in payload_by_id
        ]

        await self._save_search_log(
            request_id, query, query_variants, filters, hybrid_result, reranked_ids, results,
            latency_query_expansion_ms=latency_query_expansion_ms,
            latency_embed_query_ms=latency_embed_query_ms,
            latency_hybrid_search_ms=latency_hybrid_search_ms,
            latency_rerank_ms=latency_rerank_ms,
        )
        return SearchDiagnostics(
            dense=hybrid_result.dense, sparse=hybrid_result.sparse, fused=hybrid_result.fused,
            reranked_ids=reranked_ids, results=results, query_variants=query_variants,
        )

    async def _save_search_log(
        self,
        request_id: str,
        query: str,
        query_variants: list[str],
        filters: SearchFilters,
        hybrid_result: HybridSearchResult,
        reranked_ids: list[str],
        results: list[SearchResultChunk],
        latency_query_expansion_ms: float,
        latency_embed_query_ms: float,
        latency_hybrid_search_ms: float,
        latency_rerank_ms: float,
    ) -> None:
        """Пишет журнал поискового запроса (Этап 8). Отказ записи не должен ронять сам поиск —
        перехватывается и логируется как предупреждение (FASTAPI_PATTERNS.md, раздел 9)."""
        search_log = SearchLog(
            request_id=request_id,
            query=query,
            query_variants=query_variants,
            audience=filters.audience,
            topic=filters.topic,
            category=filters.category,
            dense_candidates=[list(item) for item in hybrid_result.dense],
            sparse_candidates=[list(item) for item in hybrid_result.sparse],
            rrf_candidates=[list(item) for item in hybrid_result.fused],
            reranked_chunk_ids=reranked_ids,
            final_response=[result.model_dump() for result in results],
            latency_query_expansion_ms=latency_query_expansion_ms,
            latency_embed_query_ms=latency_embed_query_ms,
            latency_hybrid_search_ms=latency_hybrid_search_ms,
            latency_rerank_ms=latency_rerank_ms,
        )
        try:
            await self.search_log_repository.save_search_log(search_log)
        except SearchLogRepositoryError as error:
            logger.warning('⚠️ Не удалось записать журнал поискового запроса %s. Детали: %s', request_id, error)
