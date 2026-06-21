import asyncio

from app.clients.embeddings import EmbeddingClient
from app.core.config_logger import logger
from app.ingestion.enrichment import build_embedding_text
from app.models.schemas import EmbeddedChunk, EnrichedChunk

# Как и обогащение (Этап 3) — офлайн-процесс, требует ограничения
# конкурентности к Yandex Cloud API.
EMBEDDING_CONCURRENCY = 5


async def embed_chunk(
    embedding_client: EmbeddingClient, enriched_chunk: EnrichedChunk, doc_model_uri: str
) -> EmbeddedChunk:
    """Эмбеддит один обогащённый чанк: заголовок+текст и каждый гипотетический вопрос.

    Args:
        embedding_client: Клиент Embedding API.
        enriched_chunk: Обогащённый чанк — результат Этапа 3.
        doc_model_uri: URI doc-модели эмбеддинга (для индексации, не для поиска).

    Returns:
        Чанк с векторами эмбеддингов.

    Raises:
        EmbeddingApiRequestError: Если все попытки запроса исчерпаны.
    """
    chunk_vector = await embedding_client.get_embedding(
        text=build_embedding_text(enriched_chunk), model_uri=doc_model_uri
    )
    question_vectors = [
        await embedding_client.get_embedding(text=question, model_uri=doc_model_uri)
        for question in enriched_chunk.hypothetical_questions
    ]
    return EmbeddedChunk(
        enriched_chunk=enriched_chunk,
        chunk_vector=chunk_vector,
        question_vectors=question_vectors,
    )


async def embed_chunks(
    embedding_client: EmbeddingClient, enriched_chunks: list[EnrichedChunk], doc_model_uri: str
) -> list[EmbeddedChunk]:
    """Эмбеддит список обогащённых чанков с ограничением конкурентности.

    Args:
        embedding_client: Клиент Embedding API.
        enriched_chunks: Обогащённые чанки — результат Этапа 3.
        doc_model_uri: URI doc-модели эмбеддинга.

    Returns:
        Чанки с векторами эмбеддингов в том же порядке, что входной список.

    Raises:
        EmbeddingApiRequestError: Если эмбеддинг хотя бы одного чанка не
            удался после исчерпания всех retry. `return_exceptions=True`
            (ING-4, AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — чтобы
            сообщить, какие именно `chunk_index` не получили вектор, вместо
            первой попавшейся ошибки из параллельного батча.
    """
    semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

    async def _embed_with_limit(enriched_chunk: EnrichedChunk) -> EmbeddedChunk:
        async with semaphore:
            return await embed_chunk(embedding_client, enriched_chunk, doc_model_uri)

    logger.info(
        '🤖 Эмбеддинг %d чанков (конкурентность: %d).', len(enriched_chunks), EMBEDDING_CONCURRENCY
    )
    results = await asyncio.gather(
        *(_embed_with_limit(enriched_chunk) for enriched_chunk in enriched_chunks), return_exceptions=True
    )

    failed_indices = [
        enriched_chunk.chunk.chunk_index
        for enriched_chunk, result in zip(enriched_chunks, results, strict=True)
        if isinstance(result, BaseException)
    ]
    if failed_indices:
        logger.warning('⚠️ Эмбеддинг не удался для chunk_index=%s из %d.', failed_indices, len(enriched_chunks))
        first_error = next(result for result in results if isinstance(result, BaseException))
        raise first_error

    logger.info('✅ Эмбеддинг завершён: %d чанков.', len(results))
    return list(results)
