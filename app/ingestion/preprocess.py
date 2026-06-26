import re

from app.models.metadata import Category
from app.models.schemas import Section

# Категория источника (5 значений, Этап 5.1 плана) определяет и стратегию
# извлечения структуры: нормативные акты и судебная практика размечены
# номерами статей ("Статья N") — `law`-парсер; авторские материалы — это
# markdown с заголовками разделов — `article`-парсер.
_CATEGORY_TO_STRUCTURE: dict[Category, str] = {
    'labor_code': 'law',
    'federal_law': 'law',
    'other_npa': 'npa',
    'case_law': 'npa',
    'authorial': 'article',
}

LAW_ARTICLE_PATTERN = re.compile(
    r'^Статья\s+(?P<number>\d+(?:\.\d+)?)\.?\s*(?P<title>.*)$',
    re.MULTILINE,
)
# Пронумерованные пункты верхнего уровня в постановлениях (Пленума ВС РФ,
# Правительства РФ, подзаконных актах): "1.", "2.", "10." — одно целое число,
# за которым точка, потом пробел/текст. Отрицательный lookahead (?!\d) исключает
# вложенную нумерацию "1.1.", "2.3." чтобы не разбивать их на отдельные секции.
# Вариант "1)" / "2)" добавлен как альтернатива — встречается в некоторых
# постановлениях и нормативных актах.
NPA_PARAGRAPH_PATTERN = re.compile(
    r'^(?P<number>\d+)(?:\.(?!\d)|\))\s+',
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


def extract_law_sections(document_id: str, text: str, category: Category) -> list[Section]:
    """Извлекает секции нормативного акта — по одной на каждую найденную статью.

    Args:
        document_id: Идентификатор документа-источника.
        text: Очищенный текст документа.
        category: Категория источника (раздел 3 плана) — переносится в каждую секцию.

    Returns:
        Список секций с номером и заголовком статьи как структурными метаданными.
        Текст до первой найденной статьи (преамбула) отбрасывается — он не
        привязан к конкретной статье и не несёт самостоятельной правовой нормы.
        Если в тексте не найдено ни одной "Статья N" (например, судебный акт,
        размеченный пунктами, а не статьями) — весь текст становится одной
        секцией, а не отбрасывается молча.
    """
    matches = list(LAW_ARTICLE_PATTERN.finditer(text))

    if not matches:
        return [
            Section(
                document_id=document_id,
                category=category,
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
        title = match.group('title').strip()

        sections.append(
            Section(
                document_id=document_id,
                category=category,
                section_index=section_index,
                section_number=match.group('number'),
                section_title=f"Статья {match.group('number')}" + (f'. {title}' if title else ''),
                text=section_text,
            )
        )

    return sections


def extract_article_sections(document_id: str, text: str, category: Category) -> list[Section]:
    """Извлекает секции авторской статьи — по одной на каждый markdown-заголовок.

    Args:
        document_id: Идентификатор документа-источника.
        text: Очищенный текст документа (markdown).
        category: Категория источника (раздел 3 плана) — переносится в каждую секцию.

    Returns:
        Список секций с заголовком раздела как структурными метаданными.
        Если в документе нет заголовков, весь текст становится одной секцией.
    """
    matches = list(MARKDOWN_HEADING_PATTERN.finditer(text))

    if not matches:
        return [
            Section(
                document_id=document_id,
                category=category,
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
                category=category,
                section_index=section_index,
                section_number=None,
                section_title=match.group('title').strip(),
                text=section_text,
            )
        )

    return sections


def extract_npa_sections(document_id: str, text: str, category: Category) -> list[Section]:
    """Извлекает секции НПА/судебной практики по нумерованным пунктам верхнего уровня.

    Постановления Пленума ВС РФ и Правительства РФ размечены пронумерованными
    пунктами ("1.", "2."), а не статьями — для них `extract_law_sections`
    не применим. Только верхний уровень ("1.", "2.") становится секцией;
    вложенные пункты ("1.1.") и подпункты ("а)", "б)") остаются внутри
    текста секции и разбиваются на чанки стандартным `chunk_section_text`.

    Если ни одного пункта не найдено — весь документ становится одной
    секцией (fallback, аналогично `extract_law_sections`).

    Args:
        document_id: Идентификатор документа-источника.
        text: Очищенный текст документа.
        category: Категория источника.

    Returns:
        Список секций с номером и началом текста пункта как заголовком.
    """
    matches = list(NPA_PARAGRAPH_PATTERN.finditer(text))

    if not matches:
        return [
            Section(
                document_id=document_id,
                category=category,
                section_index=0,
                section_number=None,
                section_title=document_id,
                text=text.strip(),
            )
        ]

    sections: list[Section] = []
    for section_index, match in enumerate(matches):
        start = match.start()
        end = matches[section_index + 1].start() if section_index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        number = match.group('number')
        first_line = section_text.split('\n', 1)[0].strip()
        title = first_line[:80] if len(first_line) > 80 else first_line

        sections.append(
            Section(
                document_id=document_id,
                category=category,
                section_index=section_index,
                section_number=number,
                section_title=title,
                text=section_text,
            )
        )

    return sections


def preprocess_document(document_id: str, raw_text: str, category: Category) -> list[Section]:
    """Препроцессит документ от Expert: очистка текста + извлечение структуры (Этап 1 плана).

    Args:
        document_id: Идентификатор документа-источника.
        raw_text: Исходный текст документа (PDF/MD/TXT уже декодированы в строку).
        category: Категория источника (раздел 3, Этап 5.1 плана) — определяет
            и метаданные, и стратегию извлечения структуры (см. `_CATEGORY_TO_STRUCTURE`).

    Returns:
        Список секций документа с текстом и структурными метаданными,
        готовый ко входу в Этап 2 (иерархический чанкинг).
    """
    text = clean_text(raw_text)

    structure = _CATEGORY_TO_STRUCTURE[category]
    if structure == 'law':
        return extract_law_sections(document_id, text, category)
    if structure == 'npa':
        return extract_npa_sections(document_id, text, category)
    return extract_article_sections(document_id, text, category)
