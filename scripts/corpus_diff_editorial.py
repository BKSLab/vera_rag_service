"""Печатает редакционные пометки корпуса с контекстом для ручного просмотра."""

import re
from pathlib import Path

from app.ingestion.chunking import chunk_document
from app.ingestion.enrichment import is_editorial_note_only
from app.ingestion.extract import extract_text_from_upload
from app.ingestion.preprocess import preprocess_document
from app.models.schemas import Chunk

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


def _match_context(chunk: Chunk, match: re.Match[str]) -> str:
    start = max(0, match.start() - _CONTEXT_CHARS)
    end = min(len(chunk.text), match.end() + _CONTEXT_CHARS)
    return chunk.text[start:end].replace('\n', '\\n')


def _print_inline_match(path: Path, chunk: Chunk, match: re.Match[str]) -> None:
    print(
        f'{path.relative_to(_ROOT)} [INLINE/CONTENT] chunk={chunk.chunk_index} '
        f'section={chunk.section_number or "-"} {match.start()}:{match.end()}'
    )
    print(_match_context(chunk, match))


def main() -> None:
    matches_count = 0
    note_only_chunks_count = 0
    for path in _corpus_files():
        raw_text = extract_text_from_upload(path.name, path.read_bytes())
        sections = preprocess_document(path.stem, raw_text, _category_for(path))
        chunks = chunk_document(sections, version='corpus-diff')
        note_only_chunks = [chunk for chunk in chunks if is_editorial_note_only(chunk.text)]
        content_chunks = [chunk for chunk in chunks if not is_editorial_note_only(chunk.text)]

        note_only_chunks_count += len(note_only_chunks)
        print(f'\n{path.relative_to(_ROOT)} — редакционные чанки ({len(note_only_chunks)})')
        for chunk in note_only_chunks:
            matches = list(_EDITORIAL_CANDIDATE_PATTERN.finditer(chunk.text))
            matches_count += len(matches)
            print(
                f'{path.relative_to(_ROOT)} [NOTE-ONLY] chunk={chunk.chunk_index} '
                f'section={chunk.section_number or "-"} совпадений={len(matches)}'
            )
            for match in matches:
                print(f'  {match.start()}:{match.end()} {_match_context(chunk, match)}')

        print(f'{path.relative_to(_ROOT)} — INLINE/CONTENT')
        for chunk in content_chunks:
            for match in _EDITORIAL_CANDIDATE_PATTERN.finditer(chunk.text):
                matches_count += 1
                _print_inline_match(path, chunk, match)

    print(f'\nВсего совпадений: {matches_count}')
    print(f'Всего note-only чанков: {note_only_chunks_count}')


if __name__ == '__main__':
    main()
