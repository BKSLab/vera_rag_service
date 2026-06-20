from sqlalchemy import select

from app.db.models.search_log import SearchLog
from app.repositories.search_log import SearchLogRepository


async def test_save_search_log_persists_all_fields(db_session):
    repository = SearchLogRepository(db_session)
    search_log = SearchLog(
        request_id='11111111-1111-1111-1111-111111111111',
        query='квота на инвалидов',
        audience='employer',
        topic='quota',
        source_type='law',
        dense_candidates=[['chunk-1', 0.9]],
        sparse_candidates=[['chunk-1', 5.2]],
        rrf_candidates=[['chunk-1', 0.032]],
        reranked_chunk_ids=['chunk-1'],
        final_response=[{'chunk_id': 'chunk-1', 'score': 0.032}],
        latency_embed_query_ms=12.5,
        latency_hybrid_search_ms=45.0,
        latency_rerank_ms=300.0,
    )

    await repository.save_search_log(search_log)

    saved = (await db_session.execute(select(SearchLog).where(SearchLog.request_id == search_log.request_id))).scalar_one()
    assert saved.query == 'квота на инвалидов'
    assert saved.dense_candidates == [['chunk-1', 0.9]]
    assert saved.reranked_chunk_ids == ['chunk-1']
    assert saved.created_at is not None
