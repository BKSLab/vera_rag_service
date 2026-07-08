from io import BytesIO
from pathlib import Path

import docx
import pdfplumber
from docx.document import Document
from lxml import etree
from lxml.etree import ElementTree, _Element

# ADM-6/ING-7/SEC-4 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — без
# лимитов загрузка одного файла могла бы потребовать неограниченной памяти
# (большой/специально сконструированный PDF) или CPU (разбор PDF с
# огромным числом страниц) — классический DoS-вектор парсинга файлов,
# особенно критичный при единственном worker'е сервиса.
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 МБ — с большим запасом для текстового документа
MAX_PDF_PAGES = 2000


class UnsupportedFileTypeError(ValueError):
    """Расширение загруженного файла не поддерживается (раздел "Этап 11.1" плана)."""

    def __init__(self, suffix: str):
        self.suffix = suffix
        super().__init__(f"Неподдерживаемый тип файла: {suffix!r}. Допустимо: .pdf, .docx, .md, .txt.")


class UploadTooLargeError(ValueError):
    """Загруженный файл превышает допустимый размер (ADM-6/ING-7/SEC-4)."""

    def __init__(self, size: int, max_size: int):
        self.size = size
        self.max_size = max_size
        super().__init__(f'Файл слишком большой: {size} байт — лимит {max_size} байт.')


class TooManyPdfPagesError(ValueError):
    """PDF содержит больше страниц, чем разумный верхний предел (ADM-6/ING-7/SEC-4)."""

    def __init__(self, pages: int, max_pages: int):
        self.pages = pages
        self.max_pages = max_pages
        super().__init__(f'PDF содержит {pages} страниц — лимит {max_pages}.')


def _extract_hyperlink_text(hyperlink_element: _Element) -> str:
    return ''.join(
        text_node.text or ''
        for text_node in ElementTree(hyperlink_element).xpath('.//w:t', namespaces=hyperlink_element.nsmap)
    )


def _extract_paragraph_text(paragraph_element: _Element, document: Document) -> str:
    """Текст параграфа .docx, включая гиперссылки, в исходном порядке.

    Изображения (`a:blip` внутри `w:r` без текста) намеренно пропускаются —
    без OCR. Перенесено из `FileTextParser` (`docx_format_handler.py`) без
    извлечения текста с картинок — там это делалось через LLM Vision,
    здесь не нужно (см. AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md, обсуждение
    с пользователем 2026-06-21).

    Номера "вставленных" статей нормативных актов (статья 133¹, 195²) в Word
    оформляются надстрочным индексом без отдельного символа-разделителя —
    голый текст рана даёт только цифру. Без реконструкции разделителя
    "133" + надстрочная "1" схлопываются в "1331", неотличимое от другого
    реального номера статьи (обнаружено на реальном ТК РФ 2026-07-08:
    "Статья 1331" вместо настоящей "Статья 133.1"). Если цифровой ран
    помечен как superscript и оба соседних символа — цифры, перед ним
    вставляется точка. Superscript в реальном документе оформлен ссылкой на
    именованный character-стиль (`w:rStyle`), а не прямым свойством рана —
    `run.font.superscript` его не видит, поэтому проверяется дополнительно
    и разрешённый стиль рана (`run.style.font.superscript`).
    """
    paragraph = docx.text.paragraph.Paragraph(paragraph_element, document)
    runs_and_hyperlinks: list[_Element] = ElementTree(paragraph_element).xpath(
        './w:r | ./w:hyperlink', namespaces=paragraph_element.nsmap
    )

    text = ''
    for element in runs_and_hyperlinks:
        tag = element.tag.split('}')[-1]
        if tag == 'hyperlink':
            text += _extract_hyperlink_text(element)
        else:
            run = docx.text.run.Run(element, paragraph)
            run_text = run.text or ''
            is_superscript = run.font.superscript or (run.style is not None and run.style.font.superscript)
            if is_superscript and text[-1:].isdigit() and run_text[:1].isdigit():
                text += '.'
            text += run_text
    return text


def _extract_table_text(table_element: _Element, document: Document) -> str:
    """Текст таблицы .docx — построчно, по ячейкам, как отдельные строки."""
    lines: list[str] = []
    for row in table_element.xpath('.//w:tr'):
        for cell in row.xpath('.//w:tc'):
            for child in cell.iterchildren():
                if child.tag.split('}')[-1] == 'p':
                    paragraph_text = _extract_paragraph_text(child, document)
                    if paragraph_text:
                        lines.append(paragraph_text)
    return '\n'.join(lines)


def _extract_section_headers_footers_text(section_element: _Element, document: Document) -> str:
    """Текст верхних/нижних колонтитулов секции .docx."""
    lines: list[str] = []
    for ref_tag in ('headerReference', 'footerReference'):
        for ref in section_element.xpath(f'.//w:{ref_tag}'):
            part = document.part.related_parts[ref.rId]
            part_tree: _Element = etree.fromstring(part.blob)
            for paragraph in part_tree.xpath('.//w:p', namespaces=part_tree.nsmap):
                text = ''.join(
                    t.text or '' for t in paragraph.xpath('.//w:t', namespaces=paragraph.nsmap)
                )
                if text:
                    lines.append(text)
    return '\n'.join(lines)


def extract_text_from_docx(content: bytes) -> str:
    """Извлекает текст из .docx, сохраняя порядок параграфов/таблиц/колонтитулов
    и переносы строк между ними (важно: `preprocess_document` ищет "Статья N"
    в начале строки — текст не схлопывается в одну строку, в отличие от
    исходной реализации в `FileTextParser`).
    """
    document = docx.Document(BytesIO(content))
    blocks: list[str] = []
    for block in document.element.body.iterchildren():
        tag = block.tag.split('}')[-1]
        if tag == 'p':
            text = _extract_paragraph_text(block, document)
        elif tag == 'tbl':
            text = _extract_table_text(block, document)
        elif tag == 'sectPr':
            text = _extract_section_headers_footers_text(block, document)
        else:
            continue
        if text:
            blocks.append(text)
    return '\n'.join(blocks)


def extract_text_from_upload(filename: str, content: bytes) -> str:
    """Декодирует загруженный файл (PDF/DOCX/MD/TXT) в текст — шаг перед
    препроцессингом (Этап 1), который ожидает на входе уже декодированную строку.

    Args:
        filename: Имя загруженного файла — расширение определяет способ извлечения.
        content: Содержимое файла.

    Returns:
        Извлечённый текст документа.

    Raises:
        UnsupportedFileTypeError: Если расширение файла не .pdf/.docx/.md/.txt.
        UploadTooLargeError: Если файл превышает `MAX_UPLOAD_SIZE_BYTES`.
        TooManyPdfPagesError: Если PDF содержит больше `MAX_PDF_PAGES` страниц.
    """
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise UploadTooLargeError(len(content), MAX_UPLOAD_SIZE_BYTES)

    suffix = Path(filename).suffix.lower()

    if suffix in ('.md', '.txt'):
        return content.decode('utf-8')

    if suffix == '.docx':
        return extract_text_from_docx(content)

    if suffix == '.pdf':
        with pdfplumber.open(BytesIO(content)) as pdf:
            if len(pdf.pages) > MAX_PDF_PAGES:
                raise TooManyPdfPagesError(len(pdf.pages), MAX_PDF_PAGES)
            return '\n'.join(page.extract_text() or '' for page in pdf.pages)

    raise UnsupportedFileTypeError(suffix)
