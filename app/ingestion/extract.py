from io import BytesIO
from pathlib import Path

import pdfplumber

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
        super().__init__(f"Неподдерживаемый тип файла: {suffix!r}. Допустимо: .pdf, .md, .txt.")


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
        UploadTooLargeError: Если файл превышает `MAX_UPLOAD_SIZE_BYTES`.
        TooManyPdfPagesError: Если PDF содержит больше `MAX_PDF_PAGES` страниц.
    """
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise UploadTooLargeError(len(content), MAX_UPLOAD_SIZE_BYTES)

    suffix = Path(filename).suffix.lower()

    if suffix in ('.md', '.txt'):
        return content.decode('utf-8')

    if suffix == '.pdf':
        with pdfplumber.open(BytesIO(content)) as pdf:
            if len(pdf.pages) > MAX_PDF_PAGES:
                raise TooManyPdfPagesError(len(pdf.pages), MAX_PDF_PAGES)
            return '\n'.join(page.extract_text() or '' for page in pdf.pages)

    raise UnsupportedFileTypeError(suffix)
