import pandas as pd


def recommend_popular_movies(
        train_ratings: pd.DataFrame,
        movies: pd.DataFrame,
        user_id: int,
        k: int=10,
) -> pd.DataFrame:
    """Recommend popular movies that the user has not already rated."""
    user_seen_movies = train_ratings.loc[
        train_ratings["userId"] == user_id,
        "movieId",
    ].unique()

    positive_ratings = train_ratings[train_ratings["is_positive"]]

    movie_scores = (
        positive_ratings.groupby("movieId")
        .size()
        .reset_index(name="positive_ratings")
    )

    movie_scores = movie_scores[
        ~movie_scores["movieId"].isin(user_seen_movies)
    ]

    recommendations = movie_scores.merge(movies, on="movieId", how="left")

    recommendations = recommendations.sort_values(
        ["positive_ratings", "movieId"],
        ascending=[False, True],
    )

    return recommendations[
        ["movieId", "title", "genres", "positive_ratings"]
    ].head(k).reset_index(drop=True)

