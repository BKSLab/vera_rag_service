from app.ingestion.preprocess import (
    clean_text,
    extract_article_sections,
    extract_law_sections,
    extract_npa_sections,
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


def test_clean_text_removes_part_section_chapter_headings():
    raw_text = (
        'ЧАСТЬ ПЕРВАЯ\n'
        '\n'
        'Раздел VI. Оплата и нормирование труда\n'
        '\n'
        'Глава 20. Общие положения\n'
        '\n'
        'Статья 129. Основные понятия\n'
        'Текст статьи 129.'
    )

    result = clean_text(raw_text)

    assert 'ЧАСТЬ' not in result
    assert 'Раздел VI' not in result
    assert 'Глава 20' not in result
    assert 'Статья 129. Основные понятия' in result


def test_extract_law_sections_splits_by_article_with_number_and_title():
    text = (
        'Статья 20. Право на труд\n'
        'Текст статьи 20.\n'
        'Статья 21. Квотирование рабочих мест\n'
        'Текст статьи 21.'
    )

    sections = extract_law_sections(document_id='fz-181', text=text, category='labor_code')

    assert len(sections) == 2
    assert sections[0].section_number == '20'
    assert sections[0].section_title == 'Статья 20. Право на труд'
    assert sections[0].text == 'Текст статьи 20.'
    assert sections[1].section_number == '21'
    assert sections[1].section_title == 'Статья 21. Квотирование рабочих мест'
    assert sections[1].text == 'Текст статьи 21.'


def test_extract_law_sections_discards_preamble_before_first_article():
    text = 'Преамбула закона, не привязанная к статье.\nСтатья 1. Общие положения\nТекст статьи 1.'

    sections = extract_law_sections(document_id='fz-181', text=text, category='labor_code')

    assert len(sections) == 1
    assert 'Преамбула' not in sections[0].text


def test_extract_law_sections_does_not_leak_chapter_heading_into_previous_article():
    text = clean_text(
        'Статья 128. Заголовок статьи 128\n'
        '\n'
        'Текст статьи 128.\n'
        '\n'
        'Раздел VI. Оплата и нормирование труда\n'
        '\n'
        'Глава 20. Общие положения\n'
        '\n'
        'Статья 129. Заголовок статьи 129\n'
        '\n'
        'Текст статьи 129.'
    )

    sections = extract_law_sections(document_id='tk-rf', text=text, category='labor_code')

    assert len(sections) == 2
    assert sections[0].text == 'Текст статьи 128.'
    assert 'Раздел' not in sections[0].text
    assert 'Глава' not in sections[0].text
    assert sections[1].text == 'Текст статьи 129.'


def test_extract_law_sections_merges_duplicate_article_number_instead_of_phantom_section():
    text = (
        'Статья 115. Нормальная продолжительность рабочего времени\n'
        '\n'
        'Текст статьи 115.\n'
        '\n'
        'Статья 115 изменена с 1 сентября 2024 г. — Федеральный закон от 01.01.2024 № 1-ФЗ\n'
        '\n'
        'Статья 116. Ежегодные дополнительные оплачиваемые отпуска\n'
        '\n'
        'Текст статьи 116.'
    )

    sections = extract_law_sections(document_id='tk-rf', text=text, category='labor_code')

    assert len(sections) == 2
    assert sections[0].section_number == '115'
    assert sections[0].text == 'Текст статьи 115.'
    assert sections[1].section_number == '116'
    assert sections[1].text == 'Текст статьи 116.'


def test_extract_article_sections_splits_by_markdown_headings():
    text = (
        '# Введение\n'
        'Текст введения.\n'
        '## Права соискателя\n'
        'Текст про права соискателя.'
    )

    sections = extract_article_sections(document_id='article-1', text=text, category='authorial')

    assert len(sections) == 2
    assert sections[0].section_title == 'Введение'
    assert sections[0].section_number is None
    assert sections[1].section_title == 'Права соискателя'
    assert sections[1].text == 'Текст про права соискателя.'


def test_extract_article_sections_returns_single_section_when_no_headings():
    text = 'Статья без заголовков, просто сплошной текст.'

    sections = extract_article_sections(document_id='article-2', text=text, category='authorial')

    assert len(sections) == 1
    assert sections[0].text == text
    assert sections[0].section_title == 'article-2'


def test_extract_npa_sections_splits_by_top_level_dot_paragraphs():
    text = (
        '1. Работодатель обязан соблюдать квоту для приёма на работу инвалидов.\n'
        '2. Квота устанавливается в размере от двух до четырёх процентов.\n'
        '3. Порядок исчисления квоты определяется Правительством.'
    )

    sections = extract_npa_sections(document_id='postanovlenie-1', text=text, category='other_npa')

    assert len(sections) == 3
    assert sections[0].section_number == '1'
    assert sections[1].section_number == '2'
    assert sections[2].section_number == '3'
    assert 'Работодатель обязан' in sections[0].text


def test_extract_npa_sections_does_not_split_on_nested_numbering():
    text = (
        '1. Работодатель обязан.\n'
        '1.1. Уточняющее требование.\n'
        '2. Следующий пункт.'
    )

    sections = extract_npa_sections(document_id='postanovlenie-2', text=text, category='other_npa')

    assert len(sections) == 2
    assert sections[0].section_number == '1'
    assert '1.1.' in sections[0].text
    assert sections[1].section_number == '2'


def test_extract_npa_sections_supports_closing_parenthesis_numbering():
    text = (
        '1) Первый пункт.\n'
        '2) Второй пункт.'
    )

    sections = extract_npa_sections(document_id='akt-1', text=text, category='case_law')

    assert len(sections) == 2
    assert sections[0].section_number == '1'
    assert sections[1].section_number == '2'


def test_extract_npa_sections_fallback_when_no_paragraphs_found():
    text = 'Текст постановления без нумерации, единый блок.'

    sections = extract_npa_sections(document_id='akt-2', text=text, category='case_law')

    assert len(sections) == 1
    assert sections[0].text == text
    assert sections[0].section_number is None


def test_preprocess_document_dispatches_to_npa_extractor_for_case_law():
    raw_text = '1. Разъяснить судам.\n2. При рассмотрении дел.'

    sections = preprocess_document(document_id='plenum-1', raw_text=raw_text, category='case_law')

    assert len(sections) == 2
    assert sections[0].section_number == '1'


def test_preprocess_document_dispatches_to_npa_extractor_for_other_npa():
    raw_text = '1. Утвердить прилагаемые Правила.\n2. Министерству.'

    sections = preprocess_document(document_id='postanovlenie-gov', raw_text=raw_text, category='other_npa')

    assert len(sections) == 2
    assert sections[0].section_number == '1'


def test_preprocess_document_dispatches_to_law_extractor():
    raw_text = 'Статья 1. Общие положения\nТекст.'

    sections = preprocess_document(document_id='fz-181', raw_text=raw_text, category='labor_code')

    assert len(sections) == 1
    assert sections[0].category == 'labor_code'


def test_preprocess_document_dispatches_to_article_extractor():
    raw_text = '# Заголовок\nТекст.'

    sections = preprocess_document(document_id='article-1', raw_text=raw_text, category='authorial')

    assert len(sections) == 1
    assert sections[0].category == 'authorial'


