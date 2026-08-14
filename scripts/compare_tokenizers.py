"""Сравнение sparse-токенизации на парах русских словоформ."""

import re

from app.vectorstore.sparse import _stem_sparse_token, tokenize

_WORD_PATTERN = re.compile(r'\w+', re.UNICODE)
_LEGACY_REPLACEMENTS = (
    (re.compile(r'\bст\.', re.IGNORECASE), 'статья'),
    (re.compile(r'\bтк\s+рф\b', re.IGNORECASE), 'трудовой кодекс'),
    (re.compile(r'\bфз\b', re.IGNORECASE), 'федеральный закон'),
)
_WORD_FORM_PAIRS = (
    ('инвалид', 'инвалидов'),
    ('трудовой договор', 'трудового договора'),
    ('сокращенная продолжительность', 'сокращенной продолжительности'),
    ('индивидуальная программа реабилитации', 'индивидуальной программой реабилитации'),
)


def _legacy_tokens(text: str, *, stemming: bool) -> list[str]:
    normalized = text.lower().replace('ё', 'е')
    for pattern, replacement in _LEGACY_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    tokens = _WORD_PATTERN.findall(normalized)
    return [_stem_sparse_token(token) for token in tokens] if stemming else tokens


def main() -> None:
    modes = (
        ('текущий', lambda text: _legacy_tokens(text, stemming=False)),
        ('стемминг', lambda text: _legacy_tokens(text, stemming=True)),
        ('расширенный словарь', lambda text: tokenize(text, sparse_stemming_enabled=False)),
    )
    for left, right in _WORD_FORM_PAIRS:
        print(f'\n{left!r} ↔ {right!r}')
        for mode_name, tokenizer in modes:
            left_tokens = tokenizer(left)
            right_tokens = tokenizer(right)
            intersection = sorted(set(left_tokens) & set(right_tokens))
            print(
                f'  {mode_name}: left={left_tokens}, right={right_tokens}, '
                f'intersection={intersection}'
            )


if __name__ == '__main__':
    main()
