import pytest

from app.core.settings import SearchSettings
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


def test_tokenize_expands_article_paragraph_and_ipra_abbreviations():
    assert tokenize('ст. 128, п. 5, ч. 2, абз. 3 и ИПРА') == [
        'статья',
        '128',
        'пункт',
        '5',
        'часть',
        '2',
        'абзац',
        '3',
        'и',
        'индивидуальная',
        'программа',
        'реабилитации',
    ]


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        ('инвалид', 'инвалидов'),
        ('трудовой договор', 'трудового договора'),
        ('сокращенная продолжительность', 'сокращенной продолжительности'),
        ('программа реабилитации', 'программой реабилитации'),
    ],
)
def test_stemming_gives_word_form_pairs_nonempty_token_intersection(left: str, right: str):
    assert set(tokenize(left, sparse_stemming_enabled=True)) & set(
        tokenize(right, sparse_stemming_enabled=True)
    )


def test_disabled_stemming_preserves_existing_tokens():
    assert tokenize('трудового договора', sparse_stemming_enabled=False) == [
        'трудового',
        'договора',
    ]


def test_sparse_stemming_is_disabled_by_default():
    assert SearchSettings().sparse_stemming_enabled is False


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
