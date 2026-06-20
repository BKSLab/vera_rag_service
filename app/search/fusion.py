# Константа сглаживания RRF — стандартное значение из оригинальной статьи
# (Cormack et al., 2009), нет смысла подбирать иначе без датасета для тюнинга.
RRF_SMOOTHING_K = 60


def rrf_fusion(ranked_lists: list[list[str]], k: int = RRF_SMOOTHING_K) -> list[tuple[str, float]]:
    """Объединяет несколько ранжированных списков идентификаторов в один (Этап 5).

    Reciprocal Rank Fusion: score(id) = sum(1 / (k + rank)) по всем спискам,
    где id встретился (rank — позиция, начиная с 1). Работает только с
    позициями, не с исходными scores — это и есть смысл RRF: dense (cosine)
    и sparse (BM25) дают несравнимые по масштабу scores, а ранги сопоставимы
    всегда.

    Args:
        ranked_lists: Списки идентификаторов, каждый — в порядке убывания релевантности.
        k: Константа сглаживания RRF.

    Returns:
        Список (id, score) в порядке убывания score. Идентификатор, попавший
        в несколько списков, получает сумму вкладов от каждого.
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
