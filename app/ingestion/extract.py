from io import BytesIO
from pathlib import Path

import pdfplumber


class UnsupportedFileTypeError(ValueError):
    """Расширение загруженного файла не поддерживается (раздел "Этап 11.1" плана)."""

    def __init__(self, suffix: str):
        self.suffix = suffix
        super().__init__(f"Неподдерживаемый тип файла: {suffix!r}. Допустимо: .pdf, .md, .txt.")


def extract_text_from_upload(filename: str, content: bytes) -> str:
    """Декодирует загруженный файл (PDF/MD/TXT) в текст — шаг перед препроцессингом
    (Этап 1), который ожидает на входе уже декодированную строку.

    Args:
        filename: Имя загруженного файла — расширение определяет способ извлечения.
        content: Содержимое файла.

    Returns:
        Извлечённый текст документа.

    Raises:
        UnsupportedFileTypeError: Если расширение файла не .pdf/.md/.txt.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in ('.md', '.txt'):
        return content.decode('utf-8')

    if suffix == '.pdf':
        with pdfplumber.open(BytesIO(content)) as pdf:
            return '\n'.join(page.extract_text() or '' for page in pdf.pages)

    raise UnsupportedFileTypeError(suffix)
