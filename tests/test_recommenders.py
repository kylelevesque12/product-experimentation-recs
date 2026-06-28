import pandas as pd

from src.recommenders import recommend_popular_movies


def test_recommend_popular_movies_excludes_seen_movies_and_ranks_by_popularity():
    train_ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 3, 4, 5],
            "movieId": [10, 20, 30, 30, 40, 40],
            "rating": [5.0, 4.0, 5.0, 4.5, 5.0, 4.0],
            "is_positive": [True, True, True, True, True, True],
        }
    )

    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": ["Seen A", "Seen B", "Popular A", "Popular B"],
            "genres": ["Drama", "Comedy", "Action", "Thriller"],
        }
    )

    recommendations = recommend_popular_movies(
        train_ratings=train_ratings,
        movies=movies,
        user_id=1,
        k=2,
    )

    assert recommendations["movieId"].tolist() == [30, 40]
    assert 10 not in recommendations["movieId"].tolist()
    assert 20 not in recommendations["movieId"].tolist()