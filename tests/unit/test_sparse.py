from app.vectorstore.sparse import text_to_sparse_vector, tokenize


def test_tokenize_lowercases_and_splits_on_word_boundaries():
    assert tokenize('Квота — 2 процента!') == ['квота', '2', 'процента']


def test_tokenize_normalizes_yo_to_e():
    assert tokenize('Всё о квоте') == ['все', 'о', 'квоте']


def test_tokenize_expands_common_legal_abbreviations():
    assert tokenize('ст. 21 ТК РФ и ФЗ') == [
        'статья',
        '21',
        'трудовой',
        'кодекс',
        'и',
        'федеральный',
        'закон',
    ]


def test_text_to_sparse_vector_is_empty_for_text_without_tokens():
    vector = text_to_sparse_vector('   —  !!!  ')

    assert vector.indices == []
    assert vector.values == []


def test_text_to_sparse_vector_counts_term_frequency():
    vector = text_to_sparse_vector('квота квота инвалидов')

    assert len(vector.indices) == 2
    assert sorted(vector.values) == [1.0, 2.0]


def test_text_to_sparse_vector_is_deterministic_for_same_text():
    first = text_to_sparse_vector('квота на инвалидов')
    second = text_to_sparse_vector('квота на инвалидов')

    assert first.indices == second.indices
    assert first.values == second.values
