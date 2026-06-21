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

    chunks: list[str] = []
    start = 0
    total_words = len(words)

    while start < total_words:
        current_words: list[str] = []
        current_chars = 0
        idx = start

        while idx < total_words:
            word = words[idx]
            projected_chars = current_chars + len(word) + 1
            if current_words and projected_chars // CHARS_PER_TOKEN_ESTIMATE > target_tokens:
                break
            current_words.append(word)
            current_chars = projected_chars
            idx += 1

        chunks.append(' '.join(current_words))

        if idx >= total_words:
            break

        overlap_chars_budget = overlap_tokens * CHARS_PER_TOKEN_ESTIMATE
        back = idx
        accumulated_chars = 0
        while back > start and accumulated_chars < overlap_chars_budget:
            back -= 1
            accumulated_chars += len(words[back]) + 1

        start = back if back > start else idx

    return chunks


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
        for chunk_text_value in chunk_text(section.text):
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
