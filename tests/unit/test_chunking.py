from app.ingestion.chunking import chunk_document, chunk_text, estimate_tokens
from app.models.schemas import Section


def make_words(count: int, word: str = 'слово') -> str:
    return ' '.join(f'{word}{i}' for i in range(count))


def test_estimate_tokens_uses_chars_per_token_heuristic():
    assert estimate_tokens('a' * 400) == 100


def test_chunk_text_returns_empty_list_for_empty_text():
    assert chunk_text('') == []


def test_chunk_text_returns_single_chunk_when_text_shorter_than_target():
    text = make_words(10)

    chunks = chunk_text(text, target_tokens=500, overlap_tokens=100)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_long_text_into_multiple_chunks_within_target():
    text = make_words(2000)

    chunks = chunk_text(text, target_tokens=500, overlap_tokens=100)

    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk) <= 500 + 1


def test_chunk_text_has_overlap_between_consecutive_chunks():
    text = make_words(2000)

    chunks = chunk_text(text, target_tokens=500, overlap_tokens=100)

    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    overlap = set(first_chunk_words) & set(second_chunk_words)

    assert overlap, 'соседние чанки должны иметь общие слова в зоне overlap'
    assert second_chunk_words[0] in first_chunk_words, 'второй чанк должен начинаться внутри первого'


def test_chunk_text_covers_all_words_without_gaps():
    text = make_words(2000)

    chunks = chunk_text(text, target_tokens=500, overlap_tokens=100)

    all_words_in_chunks = set(word for chunk in chunks for word in chunk.split())
    original_words = set(text.split())

    assert original_words <= all_words_in_chunks


def test_chunk_document_assigns_sequential_chunk_index_across_sections():
    sections = [
        Section(
            document_id='fz-181',
            category='labor_code',
            section_index=0,
            section_number='20',
            section_title='Статья 20',
            text=make_words(2000),
        ),
        Section(
            document_id='fz-181',
            category='labor_code',
            section_index=1,
            section_number='21',
            section_title='Статья 21',
            text=make_words(10),
        ),
    ]

    chunks = chunk_document(sections, version='2026-01-01')

    chunk_indices = [chunk.chunk_index for chunk in chunks]
    assert chunk_indices == list(range(len(chunks)))
    assert len(chunks) > 2


def test_chunk_document_preserves_section_metadata_on_each_chunk():
    sections = [
        Section(
            document_id='fz-181',
            category='labor_code',
            section_index=0,
            section_number='21',
            section_title='Статья 21. Квотирование рабочих мест',
            text=make_words(2000),
        ),
    ]

    chunks = chunk_document(sections, version='2026-01-01')

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document_id == 'fz-181'
        assert chunk.category == 'labor_code'
        assert chunk.section_index == 0
        assert chunk.section_number == '21'
        assert chunk.section_title == 'Статья 21. Квотирование рабочих мест'


def test_chunk_document_generates_unique_chunk_ids():
    sections = [
        Section(
            document_id='fz-181',
            category='labor_code',
            section_index=0,
            section_number='21',
            section_title='Статья 21',
            text=make_words(2000),
        ),
    ]

    chunks = chunk_document(sections, version='2026-01-01')

    chunk_ids = {chunk.chunk_id for chunk in chunks}
    assert len(chunk_ids) == len(chunks)


def test_chunk_document_is_deterministic_for_same_document_id_version_and_index():
    """ING-1 — повторный ingestion того же document_id+version должен давать
    те же chunk_id, чтобы Qdrant upsert перезаписывал точки, а не плодил дубли."""
    sections = [
        Section(
            document_id='fz-181',
            category='labor_code',
            section_index=0,
            section_number='21',
            section_title='Статья 21',
            text=make_words(2000),
        ),
    ]

    first_run = chunk_document(sections, version='2026-01-01')
    second_run = chunk_document(sections, version='2026-01-01')

    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]


def test_chunk_document_generates_different_chunk_ids_for_different_versions():
    sections = [
        Section(
            document_id='fz-181',
            category='labor_code',
            section_index=0,
            section_number='21',
            section_title='Статья 21',
            text=make_words(10),
        ),
    ]

    v1 = chunk_document(sections, version='2026-01-01')
    v2 = chunk_document(sections, version='2026-02-01')

    assert {chunk.chunk_id for chunk in v1}.isdisjoint({chunk.chunk_id for chunk in v2})
