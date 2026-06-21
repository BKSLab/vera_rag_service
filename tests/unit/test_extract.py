from contextlib import contextmanager
from io import BytesIO
from unittest.mock import MagicMock, patch

import docx
import pytest

from app.ingestion.extract import (
    MAX_UPLOAD_SIZE_BYTES,
    TooManyPdfPagesError,
    UnsupportedFileTypeError,
    UploadTooLargeError,
    extract_text_from_upload,
)


def make_docx_bytes(paragraphs: list[str], table_rows: list[tuple[str, str]] | None = None) -> bytes:
    document = docx.Document()
    for paragraph_text in paragraphs:
        document.add_paragraph(paragraph_text)
    if table_rows:
        table = document.add_table(rows=len(table_rows), cols=2)
        for row, (left, right) in zip(table.rows, table_rows, strict=True):
            row.cells[0].text = left
            row.cells[1].text = right
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_extract_text_from_upload_decodes_txt():
    result = extract_text_from_upload('doc.txt', 'Текст документа.'.encode())

    assert result == 'Текст документа.'


def test_extract_text_from_upload_decodes_md():
    result = extract_text_from_upload('doc.md', '# Заголовок\nТекст.'.encode())

    assert result == '# Заголовок\nТекст.'


def test_extract_text_from_upload_raises_on_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        extract_text_from_upload('doc.rtf', b'irrelevant')


def test_extract_text_from_upload_raises_when_file_too_large():
    """ADM-6/ING-7/SEC-4 — лимит размера файла, защита от DoS через
    загрузку чрезмерно большого/специально сконструированного файла."""
    oversized_content = b'a' * (MAX_UPLOAD_SIZE_BYTES + 1)

    with pytest.raises(UploadTooLargeError):
        extract_text_from_upload('doc.txt', oversized_content)


def test_extract_text_from_upload_decodes_docx_paragraphs_preserving_line_breaks():
    """Перенесено из FileTextParser (docx_format_handler.py) — текст параграфов
    в исходном порядке. Важно: переносы строк между параграфами сохраняются
    (не схлопываются в одну строку, как в исходной реализации) — иначе
    `preprocess_document` не найдёт "Статья N" в начале строки."""
    content = make_docx_bytes(['Первый параграф.', 'Статья 21. Квотирование рабочих мест', 'Текст статьи.'])

    result = extract_text_from_upload('doc.docx', content)

    assert result.splitlines() == ['Первый параграф.', 'Статья 21. Квотирование рабочих мест', 'Текст статьи.']


def test_extract_text_from_upload_decodes_docx_table_cells():
    content = make_docx_bytes(['Текст до таблицы.'], table_rows=[('Ячейка 1', 'Ячейка 2')])

    result = extract_text_from_upload('doc.docx', content)

    assert 'Ячейка 1' in result
    assert 'Ячейка 2' in result


def test_extract_text_from_upload_raises_when_pdf_has_too_many_pages():
    """ADM-6/ING-7/SEC-4 — лимит числа страниц PDF проверяется до
    извлечения текста по каждой странице (самая дорогая операция)."""
    fake_pdf = MagicMock()
    fake_pdf.pages = [MagicMock() for _ in range(2001)]

    @contextmanager
    def fake_open(_buffer):
        yield fake_pdf

    with patch('app.ingestion.extract.pdfplumber.open', fake_open), pytest.raises(TooManyPdfPagesError):
        extract_text_from_upload('doc.pdf', b'%PDF-1.4 fake content')
