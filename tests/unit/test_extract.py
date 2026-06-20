import pytest

from app.ingestion.extract import UnsupportedFileTypeError, extract_text_from_upload


def test_extract_text_from_upload_decodes_txt():
    result = extract_text_from_upload('doc.txt', 'Текст документа.'.encode('utf-8'))

    assert result == 'Текст документа.'


def test_extract_text_from_upload_decodes_md():
    result = extract_text_from_upload('doc.md', '# Заголовок\nТекст.'.encode('utf-8'))

    assert result == '# Заголовок\nТекст.'


def test_extract_text_from_upload_raises_on_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        extract_text_from_upload('doc.docx', b'irrelevant')
