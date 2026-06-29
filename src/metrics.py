import math


def precision_at_k(
    recommended_items: list[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """Calculate Precision@K for one user."""
    if k <= 0:
        raise ValueError("k must be positive.")

    top_k_items = recommended_items[:k]

    if len(top_k_items) == 0:
        return 0.0

    hits = 0

    for item in top_k_items:
        if item in relevant_items:
            hits += 1

    return hits / len(top_k_items)


def recall_at_k(
    recommended_items: list[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """Calculate Recall@K for one user."""
    if k <= 0:
        raise ValueError("k must be positive.")

    if len(relevant_items) == 0:
        return 0.0

    top_k_items = recommended_items[:k]

    hits = 0

    for item in top_k_items:
        if item in relevant_items:
            hits += 1

    return hits / len(relevant_items)


def ndcg_at_k(
    recommended_items: list[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """Calculate NDCG@K for one user."""
    if k <= 0:
        raise ValueError("k must be positive.")

    if len(relevant_items) == 0:
        return 0.0

    top_k_items = recommended_items[:k]

    dcg = 0.0

    for index, item in enumerate(top_k_items):
        rank = index + 1

        if item in relevant_items:
            dcg += 1 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)
    ideal_dcg = 0.0

    for rank in range(1, ideal_hits + 1):
        ideal_dcg += 1 / math.log2(rank + 1)

    return dcg / ideal_dcg


def rated_precision_at_k(
    recommended_items: list[int],
    positive_items: set[int],
    rated_items: set[int],
    k: int,
) -> float:
    """Calculate precision among top-K recommendations the user later rated."""
    if k <= 0:
        raise ValueError("k must be positive.")

    top_k_items = recommended_items[:k]

    rated_recommendations = []

    for item in top_k_items:
        if item in rated_items:
            rated_recommendations.append(item)

    if len(rated_recommendations) == 0:
        return float("nan")

    hits = 0

    for item in rated_recommendations:
        if item in positive_items:
            hits += 1

    return hits / len(rated_recommendations)
