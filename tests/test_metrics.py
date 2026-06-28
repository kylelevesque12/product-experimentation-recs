from src.metrics import precision_at_k, recall_at_k, ndcg_at_k


def test_precision_at_k_counts_relevant_recommendations():
    recommended_items = [10, 20, 30, 40, 50]
    relevant_items = {20, 40, 60}

    score = precision_at_k(
        recommended_items=recommended_items,
        relevant_items=relevant_items,
        k=5,
    )

    assert score == 0.4


def test_recall_at_k_counts_fraction_of_relevant_items_recovered():
    recommended_items = [10, 20, 30, 40, 50]
    relevant_items = {20, 40, 60, 80}

    score = recall_at_k(
        recommended_items=recommended_items,
        relevant_items=relevant_items,
        k=5,
    )

    assert score == 0.5


def test_ndcg_at_k_rewards_relevant_items_near_top():
    relevant_items = {10, 30}

    better_recommendations = [10, 30, 20, 40]
    worse_recommendations = [20, 40, 10, 30]

    better_score = ndcg_at_k(
        recommended_items=better_recommendations,
        relevant_items=relevant_items,
        k=4,
    )

    worse_score = ndcg_at_k(
        recommended_items=worse_recommendations,
        relevant_items=relevant_items,
        k=4,
    )

    assert better_score > worse_score
    assert better_score == 1.0

    