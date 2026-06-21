from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.extract import (
    MAX_UPLOAD_SIZE_BYTES,
    TooManyPdfPagesError,
    UnsupportedFileTypeError,
    UploadTooLargeError,
    extract_text_from_upload,
)


def test_extract_text_from_upload_decodes_txt():
    result = extract_text_from_upload('doc.txt', 'Текст документа.'.encode())

    assert result == 'Текст документа.'


def test_extract_text_from_upload_decodes_md():
    result = extract_text_from_upload('doc.md', '# Заголовок\nТекст.'.encode())

    assert result == '# Заголовок\nТекст.'


def test_extract_text_from_upload_raises_on_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        extract_text_from_upload('doc.docx', b'irrelevant')


def test_extract_text_from_upload_raises_when_file_too_large():
    """ADM-6/ING-7/SEC-4 — лимит размера файла, защита от DoS через
    загрузку чрезмерно большого/специально сконструированного файла."""
    oversized_content = b'a' * (MAX_UPLOAD_SIZE_BYTES + 1)

    with pytest.raises(UploadTooLargeError):
        extract_text_from_upload('doc.txt', oversized_content)


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
