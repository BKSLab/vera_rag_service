import re

from app.models.schemas import Section

LAW_ARTICLE_PATTERN = re.compile(
    r'^Статья\s+(?P<number>\d+(?:\.\d+)?)\.?\s*(?P<title>.*)$',
    re.MULTILINE,
)
MARKDOWN_HEADING_PATTERN = re.compile(r'^#{1,3}\s+(?P<title>.+)$', re.MULTILINE)

PAGE_FOOTER_PATTERN = re.compile(r'^\s*-?\s*\d+\s*-?\s*$', re.MULTILINE)
MULTIPLE_BLANK_LINES_PATTERN = re.compile(r'\n{3,}')
SOFT_HYPHEN_LINEBREAK_PATTERN = re.compile(r'-\n(?=\w)')


def clean_text(raw_text: str) -> str:
    """Очищает текст документа от артефактов форматирования перед извлечением структуры.

    Убирает переносы строк по дефису внутри слова (артефакт PDF-вёрстки),
    одиночные строки-колонтитулы с номером страницы и схлопывает более
    двух подряд идущих пустых строк до одной.

    Args:
        raw_text: Исходный текст документа.

    Returns:
        Очищенный текст с нормализованными пробелами и переносами строк.
    """
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    text = SOFT_HYPHEN_LINEBREAK_PATTERN.sub('', text)
    text = PAGE_FOOTER_PATTERN.sub('', text)
    text = MULTIPLE_BLANK_LINES_PATTERN.sub('\n\n', text)
    return text.strip()


def extract_law_sections(document_id: str, text: str) -> list[Section]:
    """Извлекает секции нормативного акта — по одной на каждую найденную статью.

    Args:
        document_id: Идентификатор документа-источника.
        text: Очищенный текст документа.

    Returns:
        Список секций с номером и заголовком статьи как структурными метаданными.
        Текст до первой найденной статьи (преамбула) отбрасывается — он не
        привязан к конкретной статье и не несёт самостоятельной правовой нормы.
    """
    matches = list(LAW_ARTICLE_PATTERN.finditer(text))
    sections: list[Section] = []

    for section_index, match in enumerate(matches):
        start = match.end()
        end = matches[section_index + 1].start() if section_index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        title = match.group('title').strip()

        sections.append(
            Section(
                document_id=document_id,
                source_type='law',
                section_index=section_index,
                section_number=match.group('number'),
                section_title=f"Статья {match.group('number')}" + (f'. {title}' if title else ''),
                text=section_text,
            )
        )

    return sections


def extract_article_sections(document_id: str, text: str) -> list[Section]:
    """Извлекает секции авторской статьи — по одной на каждый markdown-заголовок.

    Args:
        document_id: Идентификатор документа-источника.
        text: Очищенный текст документа (markdown).

    Returns:
        Список секций с заголовком раздела как структурными метаданными.
        Если в документе нет заголовков, весь текст становится одной секцией.
    """
    matches = list(MARKDOWN_HEADING_PATTERN.finditer(text))

    if not matches:
        return [
            Section(
                document_id=document_id,
                source_type='article',
                section_index=0,
                section_number=None,
                section_title=document_id,
                text=text.strip(),
            )
        ]

    sections: list[Section] = []
    for section_index, match in enumerate(matches):
        start = match.end()
        end = matches[section_index + 1].start() if section_index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        sections.append(
            Section(
                document_id=document_id,
                source_type='article',
                section_index=section_index,
                section_number=None,
                section_title=match.group('title').strip(),
                text=section_text,
            )
        )

    return sections


def preprocess_document(document_id: str, raw_text: str, source_type: str) -> list[Section]:
    """Препроцессит документ от Expert: очистка текста + извлечение структуры (Этап 1 плана).

    Args:
        document_id: Идентификатор документа-источника.
        raw_text: Исходный текст документа (PDF/MD/TXT уже декодированы в строку).
        source_type: 'law' для нормативных актов, 'article' для авторских статей.

    Returns:
        Список секций документа с текстом и структурными метаданными,
        готовый ко входу в Этап 2 (иерархический чанкинг).

    Raises:
        ValueError: Если source_type не 'law' и не 'article'.
    """
    text = clean_text(raw_text)

    if source_type == 'law':
        return extract_law_sections(document_id, text)
    if source_type == 'article':
        return extract_article_sections(document_id, text)

    raise ValueError(f"Неизвестный source_type: {source_type!r}. Допустимо: 'law', 'article'.")
