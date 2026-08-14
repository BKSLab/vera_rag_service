"""Печатает редакционные пометки корпуса с контекстом для ручного просмотра."""

import re
from pathlib import Path

from app.ingestion.chunking import chunk_document
from app.ingestion.enrichment import is_editorial_note_only
from app.ingestion.extract import extract_text_from_upload
from app.ingestion.preprocess import clean_text, preprocess_document

_ROOT = Path(__file__).resolve().parents[1]
_CONTEXT_CHARS = 80
_SUPPORTED_SUFFIXES = {'.docx', '.pdf', '.md', '.txt'}
_EDITORIAL_CANDIDATE_PATTERN = re.compile(
    r'\([^)]{0,300}(?:федеральн\w*\s+закон\w*|\bфз\b)[^)]{0,300}\)',
    re.IGNORECASE | re.DOTALL,
)


def _corpus_files() -> list[Path]:
    files = [_ROOT / 'Трудовой_кодекс_Российской_Федерации.docx']
    npa_root = _ROOT / 'НПА'
    files.extend(
        path
        for path in npa_root.rglob('*')
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    return files


def _category_for(path: Path) -> str:
    return 'labor_code' if path.name == 'Трудовой_кодекс_Российской_Федерации.docx' else 'other_npa'


def main() -> None:
    matches_count = 0
    for path in _corpus_files():
        raw_text = extract_text_from_upload(path.name, path.read_bytes())
        text = clean_text(raw_text)
        sections = preprocess_document(path.stem, text, _category_for(path))
        chunks = chunk_document(sections, version='corpus-diff')

        for match in _EDITORIAL_CANDIDATE_PATTERN.finditer(text):
            matches_count += 1
            owner = next((chunk for chunk in chunks if match.group(0) in chunk.text), None)
            label = 'NOTE-ONLY' if owner and is_editorial_note_only(owner.text) else 'INLINE/CONTENT'
            start = max(0, match.start() - _CONTEXT_CHARS)
            end = min(len(text), match.end() + _CONTEXT_CHARS)
            context = text[start:end].replace('\n', '\\n')
            print(f'{path.relative_to(_ROOT)} [{label}] {match.start()}:{match.end()}')
            print(context)

    print(f'\nВсего совпадений: {matches_count}')


if __name__ == '__main__':
    main()
