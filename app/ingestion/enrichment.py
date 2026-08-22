import asyncio
import re

from app.clients.llm import LlmClient
from app.core.config_logger import logger
from app.ingestion.prompts.enrichment import CHUNK_ENRICHMENT_PROMPT
from app.models.schemas import Chunk, ChunkEnrichmentResult, DocumentMetadataInput, EnrichedChunk

# Ingestion — офлайн-процесс, не в hot path поиска (RAG_SERVICE_PLAN.md,
# раздел 4 «Зависимости и риски»), но конкурентные вызовы LLM всё равно
# нужно ограничивать, чтобы не упереться в rate limit Yandex Cloud API.
ENRICHMENT_CONCURRENCY = 5

_EDITORIAL_NOTE_ONLY_PATTERN = re.compile(
    r'^\s*(?:'
    r'(?:(?:часть|пункт|абзац|статья|наименование|раздел|глава)\.\s*)?'
    r'\('
    r'(?=[^)]{0,500}(?:федеральн\w*\s+закон\w*|фз))'
    r'(?=[^)]{0,500}(?:в\s+ред\.|редакци|введен|дополнен|изменен|утратил|'
    r'исключен|наименование|статья|пункт|часть|абзац|раздел|глава))'
    r'[^)]{1,500}\)\s*'
    r')+$',
    re.IGNORECASE | re.DOTALL,
)


def is_editorial_note_only(text: str) -> bool:
    """Проверяет, состоит ли чанк целиком из редакционной пометки."""
    return _EDITORIAL_NOTE_ONLY_PATTERN.fullmatch(text) is not None


def build_index_prefix(chunk: Chunk) -> str:
    """Возвращает информативный заголовок секции для индексируемого текста."""
    title = chunk.section_title
    if not title or title == chunk.document_id:
        return ''
    if chunk.text.lstrip().startswith(title[:40]):
        return ''
    return title


async def enrich_chunk(
    llm_client: LlmClient,
    chunk: Chunk,
    document_metadata: DocumentMetadataInput,
) -> EnrichedChunk:
    """Обогащает один чанк синтетическим заголовком и гипотетическими вопросами.

    Args:
        llm_client: Клиент LLM (Polza AI/Gemini, см. `get_enrichment_llm_client`).
        chunk: Чанк — результат Этапа 2.

    Returns:
        Чанк, дополненный синтетическим заголовком и 3–5 гипотетическими вопросами.

    Raises:
        LlmApiRequestError: Если все попытки запроса к LLM исчерпаны.
    """
    # Тег-разделитель (LLM-3/SEC-5, AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md)
    # — текст документа — недоверенный вход (готовится Expert, но риск
    # prompt injection не зависит от текущего уровня доверия к источнику).
    result: ChunkEnrichmentResult = await llm_client.get_llm_response(
        content=(
            f'<source>{document_metadata.source_title}</source>\n'
            f'<section>{chunk.section_title}</section>\n'
            f'<document_text>{chunk.text}</document_text>'
        ),
        prompt=CHUNK_ENRICHMENT_PROMPT,
        schema=ChunkEnrichmentResult,
    )
    return EnrichedChunk(
        chunk=chunk,
        synthetic_title=result.synthetic_title,
        hypothetical_questions=result.hypothetical_questions,
    )


async def enrich_chunks(
    llm_client: LlmClient,
    chunks: list[Chunk],
    document_metadata: DocumentMetadataInput,
) -> list[EnrichedChunk]:
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
            гипотетических вопросов в индексе (осознанное решение, см.
            RAG_SERVICE_PLAN.md, раздел 7, ING-4). Используем
            `return_exceptions=True`, чтобы при отказе сообщить, какие именно
            `chunk_index` не обогатились, а не просто первую попавшуюся
            ошибку из параллельного батча.
    """
    semaphore = asyncio.Semaphore(ENRICHMENT_CONCURRENCY)

    async def _enrich_with_limit(chunk: Chunk) -> EnrichedChunk:
        async with semaphore:
            return await enrich_chunk(llm_client, chunk, document_metadata)

    logger.info('🤖 Обогащение %d чанков (конкурентность: %d).', len(chunks), ENRICHMENT_CONCURRENCY)
    results = await asyncio.gather(*(_enrich_with_limit(chunk) for chunk in chunks), return_exceptions=True)

    failed_indices = [chunk.chunk_index for chunk, result in zip(chunks, results, strict=True) if isinstance(result, BaseException)]
    if failed_indices:
        logger.warning('⚠️ Обогащение не удалось для chunk_index=%s из %d.', failed_indices, len(chunks))
        first_error = next(result for result in results if isinstance(result, BaseException))
        raise first_error

    logger.info('✅ Обогащение завершено: %d чанков.', len(results))
    return list(results)


def build_embedding_text(enriched_chunk: EnrichedChunk) -> str:
    """Формирует текст для эмбеддинга (Этап 4): заголовок + текст чанка.

    Гипотетические вопросы эмбеддятся отдельно от текста чанка как
    дополнительные векторы (RAG_SERVICE_PLAN.md, Этап 4) — не входят сюда.

    Args:
        enriched_chunk: Обогащённый чанк.

    Returns:
        Текст вида "<synthetic_title>\\n\\n<текст чанка>".
    """
    chunk = enriched_chunk.chunk
    parts = [build_index_prefix(chunk), enriched_chunk.synthetic_title, chunk.text]
    return '\n\n'.join(part for part in parts if part)
