import pytest

from app.ingestion.preprocess import (
    clean_text,
    extract_article_sections,
    extract_law_sections,
    preprocess_document,
)


def test_clean_text_removes_hyphenated_linebreak_inside_word():
    raw_text = 'инвали-\nдов труда'

    result = clean_text(raw_text)

    assert result == 'инвалидов труда'


def test_clean_text_removes_page_footer_line():
    raw_text = 'Текст статьи.\n42\nПродолжение текста.'

    result = clean_text(raw_text)

    assert '42' not in result


def test_clean_text_collapses_multiple_blank_lines():
    raw_text = 'Первый абзац.\n\n\n\n\nВторой абзац.'

    result = clean_text(raw_text)

    assert result == 'Первый абзац.\n\nВторой абзац.'


def test_extract_law_sections_splits_by_article_with_number_and_title():
    text = (
        'Статья 20. Право на труд\n'
        'Текст статьи 20.\n'
        'Статья 21. Квотирование рабочих мест\n'
        'Текст статьи 21.'
    )

    sections = extract_law_sections(document_id='fz-181', text=text)

    assert len(sections) == 2
    assert sections[0].section_number == '20'
    assert sections[0].section_title == 'Статья 20. Право на труд'
    assert sections[0].text == 'Текст статьи 20.'
    assert sections[1].section_number == '21'
    assert sections[1].section_title == 'Статья 21. Квотирование рабочих мест'
    assert sections[1].text == 'Текст статьи 21.'


def test_extract_law_sections_discards_preamble_before_first_article():
    text = 'Преамбула закона, не привязанная к статье.\nСтатья 1. Общие положения\nТекст статьи 1.'

    sections = extract_law_sections(document_id='fz-181', text=text)

    assert len(sections) == 1
    assert 'Преамбула' not in sections[0].text


def test_extract_article_sections_splits_by_markdown_headings():
    text = (
        '# Введение\n'
        'Текст введения.\n'
        '## Права соискателя\n'
        'Текст про права соискателя.'
    )

    sections = extract_article_sections(document_id='article-1', text=text)

    assert len(sections) == 2
    assert sections[0].section_title == 'Введение'
    assert sections[0].section_number is None
    assert sections[1].section_title == 'Права соискателя'
    assert sections[1].text == 'Текст про права соискателя.'


def test_extract_article_sections_returns_single_section_when_no_headings():
    text = 'Статья без заголовков, просто сплошной текст.'

    sections = extract_article_sections(document_id='article-2', text=text)

    assert len(sections) == 1
    assert sections[0].text == text
    assert sections[0].section_title == 'article-2'


def test_preprocess_document_dispatches_to_law_extractor():
    raw_text = 'Статья 1. Общие положения\nТекст.'

    sections = preprocess_document(document_id='fz-181', raw_text=raw_text, source_type='law')

    assert len(sections) == 1
    assert sections[0].source_type == 'law'


def test_preprocess_document_dispatches_to_article_extractor():
    raw_text = '# Заголовок\nТекст.'

    sections = preprocess_document(document_id='article-1', raw_text=raw_text, source_type='article')

    assert len(sections) == 1
    assert sections[0].source_type == 'article'


def test_preprocess_document_raises_on_unknown_source_type():
    with pytest.raises(ValueError):
        preprocess_document(document_id='doc-1', raw_text='текст', source_type='unknown')
