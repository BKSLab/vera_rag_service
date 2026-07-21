"""Считает количество статей/секций в документе через реальный
preprocess-пайплайн проекта — без единого вызова LLM/embedding API.

Запуск из корня репозитория vera_rag_service:
    venv/Scripts/python.exe count_articles.py "Трудовой_кодекс_Российской_Федерации.docx"
    venv/Scripts/python.exe count_articles.py путь/к/файлу.docx labor_code
"""
import sys
from pathlib import Path

sys.path.insert(0, '.')

from app.ingestion.extract import extract_text_from_upload  # noqa: E402
from app.ingestion.preprocess import preprocess_document  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('Использование: count_articles.py <путь_к_файлу> [категория=labor_code]')
        sys.exit(1)

    file_path = Path(sys.argv[1])
    category = sys.argv[2] if len(sys.argv) > 2 else 'labor_code'

    content = file_path.read_bytes()
    raw_text = extract_text_from_upload(file_path.name, content)
    sections = preprocess_document(file_path.stem, raw_text, category)

    numbers = [s.section_number for s in sections]
    duplicates = {n for n in numbers if numbers.count(n) > 1}
    with_dot = [n for n in numbers if n and '.' in n]
    suspicious_no_dot = [n for n in numbers if n and '.' not in n and len(n) > 3]

    print(f'Файл: {file_path.name}')
    print(f'Длина текста: {len(raw_text)} символов')
    print(f'Секций (статей): {len(sections)}')
    print(f'  из них с точкой в номере (N.M): {len(with_dot)}')
    print(f'  подозрительных "слипшихся" номеров (4+ цифр без точки): {len(suspicious_no_dot)} {suspicious_no_dot[:20]}')
    print(f'  дублирующихся номеров: {len(duplicates)} {sorted(duplicates)}')


if __name__ == '__main__':
    main()
