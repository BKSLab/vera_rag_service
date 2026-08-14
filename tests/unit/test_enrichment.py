from unittest.mock import AsyncMock

from app.clients.llm import LlmClient
from app.ingestion.chunking import chunk_document
from app.ingestion.enrichment import build_embedding_text, build_index_prefix, enrich_chunk, enrich_chunks
from app.ingestion.preprocess import preprocess_document
from app.models.schemas import Chunk, ChunkEnrichmentResult, EnrichedChunk
from app.vectorstore.sparse import tokenize


def make_chunk(chunk_index: int = 0, text: str = 'Текст чанка') -> Chunk:
    return Chunk(
        chunk_id=f'chunk-{chunk_index}',
        chunk_index=chunk_index,
        chunk_number_in_section=0,
        document_id='fz-181',
        parent_id='fz-181:21',
        category='labor_code',
        section_index=0,
        section_number='21',
        section_title='Статья 21',
        text=text,
    )


async def test_enrich_chunk_returns_enriched_chunk_with_llm_result():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = ChunkEnrichmentResult(
        synthetic_title='Квотирование рабочих мест',
        hypothetical_questions=['Вопрос 1?', 'Вопрос 2?', 'Вопрос 3?'],
    )
    chunk = make_chunk()

    result = await enrich_chunk(llm_client, chunk)

    assert isinstance(result, EnrichedChunk)
    assert result.chunk == chunk
    assert result.synthetic_title == 'Квотирование рабочих мест'
    assert len(result.hypothetical_questions) == 3


async def test_enrich_chunk_passes_chunk_text_and_schema_to_llm_client():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = ChunkEnrichmentResult(
        synthetic_title='Заголовок',
        hypothetical_questions=['В1?', 'В2?', 'В3?'],
    )
    chunk = make_chunk(text='Конкретный текст статьи закона.')

    await enrich_chunk(llm_client, chunk)

    _, call_kwargs = llm_client.get_llm_response.call_args
    assert call_kwargs['content'] == '<document_text>Конкретный текст статьи закона.</document_text>'
    assert call_kwargs['schema'] is ChunkEnrichmentResult


async def test_enrich_chunks_preserves_order_for_multiple_chunks():
    llm_client = AsyncMock(spec=LlmClient)

    async def fake_response(content: str, prompt: str, schema=None, **kwargs):
        return ChunkEnrichmentResult(
            synthetic_title=f'Заголовок для: {content}',
            hypothetical_questions=['В1?', 'В2?', 'В3?'],
        )

    llm_client.get_llm_response.side_effect = fake_response
    chunks = [make_chunk(chunk_index=i, text=f'Текст {i}') for i in range(7)]

    enriched = await enrich_chunks(llm_client, chunks)

    assert len(enriched) == 7
    for i, enriched_chunk in enumerate(enriched):
        assert enriched_chunk.chunk.chunk_index == i
        assert enriched_chunk.synthetic_title == f'Заголовок для: <document_text>Текст {i}</document_text>'


def test_build_embedding_text_combines_title_and_chunk_text():
    enriched_chunk = EnrichedChunk(
        chunk=make_chunk(text='Текст статьи 21.'),
        synthetic_title='Квотирование рабочих мест',
        hypothetical_questions=['В1?', 'В2?', 'В3?'],
    )

    result = build_embedding_text(enriched_chunk)

    assert result == 'Статья 21\n\nКвотирование рабочих мест\n\nТекст статьи 21.'


def test_article_number_is_searchable_in_index_text_after_preprocess_and_chunking():
    sections = preprocess_document(
        document_id='tk-rf',
        raw_text='Статья 128. Отпуск без сохранения заработной платы\nРаботнику предоставляется отпуск.',
        category='labor_code',
    )
    chunk = chunk_document(sections, version='2026-01-01')[0]

    tokens = tokenize(f'{build_index_prefix(chunk)}\nТК РФ\n{chunk.text}')

    assert 'статья' in tokens
    assert '128' in tokens


def test_build_index_prefix_is_empty_when_case_law_title_duplicates_text_start():
    chunk = make_chunk(text='1. При рассмотрении дел о восстановлении на работе суд учитывает обстоятельства.')
    chunk.category = 'case_law'
    chunk.section_title = '1. При рассмотрении дел о восстановлении на работе суд учитывает'

    assert build_index_prefix(chunk) == ''


def test_build_index_prefix_is_empty_for_fallback_section_title():
    chunk = make_chunk()
    chunk.section_title = chunk.document_id

    assert build_index_prefix(chunk) == ''


def test_build_index_prefix_keeps_meaningful_authorial_heading():
    chunk = make_chunk(text='Практические рекомендации по оформлению документов.')
    chunk.category = 'authorial'
    chunk.section_number = None
    chunk.section_title = 'Как оформить квоту для инвалидов'

    assert build_index_prefix(chunk) == 'Как оформить квоту для инвалидов'
