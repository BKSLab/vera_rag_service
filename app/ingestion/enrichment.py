import asyncio

from app.clients.llm import LlmClient
from app.core.config_logger import logger
from app.ingestion.prompts.enrichment import CHUNK_ENRICHMENT_PROMPT
from app.models.schemas import Chunk, ChunkEnrichmentResult, EnrichedChunk

# Ingestion — офлайн-процесс, не в hot path поиска (RAG_SERVICE_PLAN.md,
# раздел 4 «Зависимости и риски»), но конкурентные вызовы LLM всё равно
# нужно ограничивать, чтобы не упереться в rate limit Yandex Cloud API.
ENRICHMENT_CONCURRENCY = 5


async def enrich_chunk(llm_client: LlmClient, chunk: Chunk) -> EnrichedChunk:
    """Обогащает один чанк синтетическим заголовком и гипотетическими вопросами.

    Args:
        llm_client: Клиент LLM (Yandex Cloud Model Gallery, см. раздел 0.1 плана).
        chunk: Чанк — результат Этапа 2.

    Returns:
        Чанк, дополненный синтетическим заголовком и 3–5 гипотетическими вопросами.

    Raises:
        LlmApiRequestError: Если все попытки запроса к LLM исчерпаны.
    """
    result: ChunkEnrichmentResult = await llm_client.get_llm_response(
        content=chunk.text,
        prompt=CHUNK_ENRICHMENT_PROMPT,
        schema=ChunkEnrichmentResult,
    )
    return EnrichedChunk(
        chunk=chunk,
        synthetic_title=result.synthetic_title,
        hypothetical_questions=result.hypothetical_questions,
    )


async def enrich_chunks(llm_client: LlmClient, chunks: list[Chunk]) -> list[EnrichedChunk]:
    """Обогащает список чанков с ограничением конкурентности к LLM API.

    Args:
        llm_client: Клиент LLM.
        chunks: Чанки — результат Этапа 2.

    Returns:
        Обогащённые чанки в том же порядке, что входной список.

    Raises:
        LlmApiRequestError: Если обогащение хотя бы одного чанка не удалось
            после исчерпания всех retry — батч не продолжается частично,
            потому что отсутствие обогащения у части корпуса важнее
            заметить на этапе ingestion, а не молча получить чанк без
            гипотетических вопросов в индексе.
    """
    semaphore = asyncio.Semaphore(ENRICHMENT_CONCURRENCY)

    async def _enrich_with_limit(chunk: Chunk) -> EnrichedChunk:
        async with semaphore:
            return await enrich_chunk(llm_client, chunk)

    logger.info('🤖 Обогащение %d чанков (конкурентность: %d).', len(chunks), ENRICHMENT_CONCURRENCY)
    enriched = await asyncio.gather(*(_enrich_with_limit(chunk) for chunk in chunks))
    logger.info('✅ Обогащение завершено: %d чанков.', len(enriched))
    return list(enriched)


def build_embedding_text(enriched_chunk: EnrichedChunk) -> str:
    """Формирует текст для эмбеддинга (Этап 4): заголовок + текст чанка.

    Гипотетические вопросы эмбеддятся отдельно от текста чанка как
    дополнительные векторы (RAG_SERVICE_PLAN.md, Этап 4) — не входят сюда.

    Args:
        enriched_chunk: Обогащённый чанк.

    Returns:
        Текст вида "<synthetic_title>\\n\\n<текст чанка>".
    """
    return f'{enriched_chunk.synthetic_title}\n\n{enriched_chunk.chunk.text}'
