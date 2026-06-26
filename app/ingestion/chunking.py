import re
from uuid import NAMESPACE_URL, uuid5

from app.models.schemas import Chunk, Section

# Неймспейс для детерминированных chunk_id (ING-1,
# AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — `chunk_id` вычисляется из
# (document_id, version, chunk_index), не генерируется случайно при каждом
# вызове. Повторный ingestion того же document_id+version естественным
# образом перезапишет те же точки Qdrant (upsert по тому же id), а не
# создаст дубликаты.
_CHUNK_ID_NAMESPACE = 'vera-rag-service'


def _deterministic_chunk_id(document_id: str, version: str, chunk_index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f'{_CHUNK_ID_NAMESPACE}:{document_id}:{version}:{chunk_index}'))

# Оценка токенов по эвристике "1 токен ~ 4 символа русского текста"
# (RAG_SERVICE_PLAN.md, раздел 3.1) — без подключения токенизатора
# конкретной embedding-модели, которая на момент чанкинга ещё не выбрана
# окончательно для прогона по реальному корпусу.
CHARS_PER_TOKEN_ESTIMATE = 4
CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100


def estimate_tokens(text: str) -> int:
    """Грубо оценивает количество токенов в тексте.

    Args:
        text: Произвольный текст.

    Returns:
        Оценка количества токенов (округление вниз, минимум 0 для пустой строки).
    """
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def chunk_text(
    text: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Разбивает текст на чанки по границам слов с overlap.

    Накопление идёт по словам до достижения целевого количества токенов
    (оценка по `estimate_tokens`), следующий чанк начинается с overlap —
    последних слов предыдущего чанка в пределах `overlap_tokens`.

    Args:
        text: Текст секции для разбиения.
        target_tokens: Целевой размер чанка в токенах.
        overlap_tokens: Размер overlap между соседними чанками в токенах.

    Returns:
        Список текстов чанков. Пустой список для пустого текста.
    """
    words = text.split()
    if not words:
        return []

    ranges: list[tuple[int, int]] = []
    start = 0
    total_words = len(words)

    while start < total_words:
        current_chars = 0
        idx = start

        while idx < total_words:
            word = words[idx]
            projected_chars = current_chars + len(word) + 1
            if idx > start and projected_chars // CHARS_PER_TOKEN_ESTIMATE > target_tokens:
                break
            current_chars = projected_chars
            idx += 1

        ranges.append((start, idx))

        if idx >= total_words:
            break

        overlap_chars_budget = overlap_tokens * CHARS_PER_TOKEN_ESTIMATE
        back = idx
        accumulated_chars = 0
        while back > start and accumulated_chars < overlap_chars_budget:
            back -= 1
            accumulated_chars += len(words[back]) + 1

        start = back if back > start else idx

    # Хвост секции после overlap-сдвига иногда оказывается короче самого
    # overlap — итоговый чанк почти целиком повторяет конец предыдущего, без
    # сколько-нибудь значимого нового текста (найдено на реальном корпусе
    # ТК РФ, 2026-06-21: статья 216.1 дала такой чанк-дубль из 543 символов,
    # начинающийся посередине предложения). Склеиваем такой хвост с
    # предыдущим чанком, а не плодим почти-дубликат отдельной точкой.
    min_new_chars = (target_tokens // 3) * CHARS_PER_TOKEN_ESTIMATE
    merged_ranges: list[tuple[int, int]] = [ranges[0]]
    for range_start, range_end in ranges[1:]:
        _, previous_end = merged_ranges[-1]
        new_chars = sum(len(word) + 1 for word in words[previous_end:range_end])
        if new_chars < min_new_chars:
            merged_ranges[-1] = (merged_ranges[-1][0], range_end)
        else:
            merged_ranges.append((range_start, range_end))

    return [' '.join(words[range_start:range_end]) for range_start, range_end in merged_ranges]


_PARAGRAPH_SPLIT_PATTERN = re.compile(r'\n\s*\n+')


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in _PARAGRAPH_SPLIT_PATTERN.split(text) if paragraph.strip()]


def _split_lines(text: str) -> list[str]:
    return [line.strip() for line in text.split('\n') if line.strip()]


def _atomic_units(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Разбивает текст секции на неделимые далее текстовые единицы.

    Иерархия (раздел 8 плана, обсуждение 2026-06-22): абзац (разделитель —
    пустая строка) → строка (одиночный перенос строки — для нормативных
    текстов это и есть граница пункта/подпункта, см. реальный `.docx` ТК РФ,
    где "1)"/"а)" уже на отдельных строках) → слова с overlap, только если
    единица текста сама по себе не укладывается в `target_tokens` даже на
    уровне одной строки. Категория не используется напрямую — деление по
    статьям/пунктам/абзацам уже заложено в том, как `Section.text` нарезан
    на Этапе 1 (см. `preprocess.py`), здесь только разбиение уже одной
    секции на единицы для упаковки в чанки.
    """
    if estimate_tokens(text) <= target_tokens:
        return [text.strip()] if text.strip() else []

    paragraphs = _split_paragraphs(text)
    if len(paragraphs) > 1:
        units: list[str] = []
        for paragraph in paragraphs:
            units.extend(_atomic_units(paragraph, target_tokens, overlap_tokens))
        return units

    lines = _split_lines(text)
    if len(lines) > 1:
        units = []
        for line in lines:
            units.extend(_atomic_units(line, target_tokens, overlap_tokens))
        return units

    # Единственная строка/абзац сама по себе превышает целевой размер —
    # последний fallback, единственное место, где применяется overlap.
    return chunk_text(text, target_tokens, overlap_tokens)


def _pack_units(units: list[str], target_tokens: int) -> list[str]:
    """Упаковывает соседние неделимые единицы в чанки до целевого размера.

    Никогда не объединяет единицы из разных вызовов (то есть из разных
    секций) — вызывающая сторона передаёт единицы только одной секции.
    Единицы, уже пришедшие из overlap-fallback `_atomic_units` (готовые
    чанки около целевого размера), не дробятся дальше — они просто не
    проходят порог для добавления соседей.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > target_tokens:
            chunks.append('\n\n'.join(current))
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


def chunk_section_text(
    text: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Разбивает текст секции (статья/пункт верхнего уровня/markdown-раздел) на чанки.

    Граница секции никогда не пересекается — функция вызывается отдельно на
    каждую секцию (раздел 8 плана, обсуждение 2026-06-22). Соседние мелкие
    единицы (строки/абзацы) одной секции упаковываются в один чанк до
    `target_tokens`; overlap применяется только при разбиении единицы,
    которая сама по себе превышает целевой размер (см. `_atomic_units`).

    Args:
        text: Текст секции.
        target_tokens: Целевой размер чанка в токенах.
        overlap_tokens: Размер overlap для словарного fallback-разбиения.

    Returns:
        Список текстов чанков. Пустой список для пустого текста.
    """
    units = _atomic_units(text, target_tokens, overlap_tokens)
    return _pack_units(units, target_tokens)


def chunk_document(sections: list[Section], version: str) -> list[Chunk]:
    """Разбивает все секции документа на чанки (Этап 2 плана).

    `chunk_index` — сквозной по всему документу, а не по отдельной секции,
    чтобы порядок чанков был восстановим при последующем upsert в Qdrant.

    Args:
        sections: Секции документа — результат Этапа 1 (препроцессинг).
        version: Версия документа (ING-1) — часть детерминированного
            `chunk_id`, чтобы повторный ingestion той же версии перезаписывал
            те же точки Qdrant, а не плодил дубликаты.

    Returns:
        Список чанков со сквозной нумерацией и метаданными секции-источника.
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        for chunk_text_value in chunk_section_text(section.text):
            chunks.append(
                Chunk(
                    chunk_id=_deterministic_chunk_id(section.document_id, version, chunk_index),
                    chunk_index=chunk_index,
                    document_id=section.document_id,
                    category=section.category,
                    section_index=section.section_index,
                    section_number=section.section_number,
                    section_title=section.section_title,
                    text=chunk_text_value,
                )
            )
            chunk_index += 1

    return chunks
