import pandas as pd

from src.data import clean_ratings, temporal_train_test_split


def test_clean_ratings_adds_datetime_and_positive_flag():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 20, 30],
            "rating": [5.0, 3.5, 4.0],
            "timestamp": [0, 60, 120],
        }
    )

    cleaned = clean_ratings(ratings)

    assert "rated_at" in cleaned.columns
    assert "is_positive" in cleaned.columns
    assert cleaned["is_positive"].tolist() == [True, False, True]


def test_temporal_train_test_split_uses_later_rows_for_test():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2],
            "movieId": [10, 20, 30, 40, 50],
            "rating": [5.0, 4.0, 3.0, 4.5, 2.0],
            "timestamp": [100, 200, 300, 100, 200],
        }
    )

    cleaned = clean_ratings(ratings)
    train, test = temporal_train_test_split(cleaned, test_fraction=0.4)

    assert set(test["movieId"]) == {30, 50}

    for user_id in test["userId"].unique():
        user_train = train[train["userId"] == user_id]
        user_test = test[test["userId"] == user_id]

        assert user_train["rated_at"].max() < user_test["rated_at"].min()