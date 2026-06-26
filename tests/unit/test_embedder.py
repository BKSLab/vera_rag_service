from unittest.mock import AsyncMock

from app.clients.embeddings import EmbeddingClient
from app.embeddings.embedder import embed_chunk, embed_chunks
from app.models.schemas import Chunk, EmbeddedChunk, EnrichedChunk


def make_enriched_chunk(chunk_index: int = 0, questions: int = 3) -> EnrichedChunk:
    chunk = Chunk(
        chunk_id=f'chunk-{chunk_index}',
        chunk_index=chunk_index,
        chunk_number_in_section=0,
        document_id='fz-181',
        parent_id='fz-181:21',
        category='labor_code',
        section_index=0,
        section_number='21',
        section_title='Статья 21',
        text='Текст чанка.',
    )
    return EnrichedChunk(
        chunk=chunk,
        synthetic_title='Заголовок',
        hypothetical_questions=[f'Вопрос {i}?' for i in range(questions)],
    )


async def test_embed_chunk_returns_chunk_vector_and_question_vectors():
    embedding_client = AsyncMock(spec=EmbeddingClient)
    embedding_client.get_embedding.return_value = [0.1, 0.2]
    enriched_chunk = make_enriched_chunk(questions=3)

    result = await embed_chunk(embedding_client, enriched_chunk, doc_model_uri='emb://folder/doc/latest')

    assert isinstance(result, EmbeddedChunk)
    assert result.chunk_vector == [0.1, 0.2]
    assert len(result.question_vectors) == 3
    assert embedding_client.get_embedding.call_count == 4


async def test_embed_chunk_uses_doc_model_uri_for_all_calls():
    embedding_client = AsyncMock(spec=EmbeddingClient)
    embedding_client.get_embedding.return_value = [0.1]
    enriched_chunk = make_enriched_chunk(questions=2)

    await embed_chunk(embedding_client, enriched_chunk, doc_model_uri='emb://folder/doc/latest')

    for call in embedding_client.get_embedding.call_args_list:
        assert call.kwargs['model_uri'] == 'emb://folder/doc/latest'


async def test_embed_chunks_preserves_order_for_multiple_chunks():
    embedding_client = AsyncMock(spec=EmbeddingClient)

    async def fake_embedding(text: str, model_uri: str):
        return [float(len(text))]

    embedding_client.get_embedding.side_effect = fake_embedding
    enriched_chunks = [make_enriched_chunk(chunk_index=i, questions=3) for i in range(6)]

    embedded = await embed_chunks(embedding_client, enriched_chunks, doc_model_uri='emb://folder/doc/latest')

    assert len(embedded) == 6
    for i, embedded_chunk in enumerate(embedded):
        assert embedded_chunk.enriched_chunk.chunk.chunk_index == i
