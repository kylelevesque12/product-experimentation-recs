import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def build_user_genre_profile(
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
) -> pd.DataFrame:
    """Count genres from a user's positive training ratings."""
    user_positive_ratings = train_ratings[
        (train_ratings["userId"] == user_id)
        & (train_ratings["is_positive"])
    ]

    user_positive_movies = user_positive_ratings.merge(
        movies,
        on="movieId",
        how="left",
    )

    genre_counts = (
        user_positive_movies["genres"]
        .str.split("|")
        .explode()
        .value_counts()
        .reset_index()
    )

    genre_counts.columns = ["genre", "positive_count"]

    return genre_counts


def recommend_genre_personalized_movies(
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    k: int = 10,
) -> pd.DataFrame:
    """Recommend unseen movies based on a user's positive genre history."""
    genre_profile = build_user_genre_profile(
        train_ratings=train_ratings,
        movies=movies,
        user_id=user_id,
    )

    # If the user has no positive ratings, fall back to recommending popular movies.
    if genre_profile.empty:
        return recommend_popular_movies(
            train_ratings=train_ratings,
            movies=movies,
            user_id=user_id,
            k=k,
        )

    genre_weights = dict(
        zip(genre_profile["genre"], genre_profile["positive_count"])
    )

    user_seen_movies = train_ratings.loc[
        train_ratings["userId"] == user_id,
        "movieId",
    ].unique()

    positive_ratings = train_ratings[train_ratings["is_positive"]]

    popularity_scores = (
        positive_ratings.groupby("movieId")
        .size()
        .reset_index(name="positive_ratings")
    )

    candidates = movies[
        ~movies["movieId"].isin(user_seen_movies)
    ].copy()

    candidates = candidates.merge(
        popularity_scores,
        on="movieId",
        how="left",
    )

    candidates["positive_ratings"] = candidates["positive_ratings"].fillna(0)

    def score_movie_genres(genres: str) -> int:
        score = 0

        for genre in genres.split("|"):
            score += genre_weights.get(genre, 0)

        return score

    candidates["genre_score"] = candidates["genres"].apply(score_movie_genres)

    recommendations = candidates.sort_values(
        ["genre_score", "positive_ratings", "movieId"],
        ascending=[False, False, True],
    )

    recommendations["positive_ratings"] = recommendations["positive_ratings"].astype(int)

    return recommendations[
        ["movieId", "title", "genres", "genre_score", "positive_ratings"]
    ].head(k).reset_index(drop=True)


def recommend_hybrid_genre_popularity_movies(
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    k: int = 10,
    genre_weight: float = 0.6,
    popularity_weight: float = 0.4,
) -> pd.DataFrame:
    """Recommend unseen movies using user genre preferences and movie popularity."""
    user_seen_movies = train_ratings.loc[
        train_ratings["userId"] == user_id,
        "movieId",
    ].unique()

    user_positive_ratings = train_ratings[
        (train_ratings["userId"] == user_id)
        & (train_ratings["is_positive"])
    ]

    if user_positive_ratings.empty:
        return recommend_popular_movies(
            train_ratings=train_ratings,
            movies=movies,
            user_id=user_id,
            k=k,
        )

    user_positive_movies = user_positive_ratings.merge(
        movies,
        on="movieId",
        how="left",
    )

    genre_weights = {}

    for genres in user_positive_movies["genres"].dropna():
        movie_genres = genres.split("|")
        genre_credit = 1 / len(movie_genres)

        for genre in movie_genres:
            genre_weights[genre] = genre_weights.get(genre, 0) + genre_credit

    total_genre_weight = sum(genre_weights.values())

    for genre in genre_weights:
        genre_weights[genre] = genre_weights[genre] / total_genre_weight

    positive_ratings = train_ratings[train_ratings["is_positive"]]

    popularity_scores = (
        positive_ratings.groupby("movieId")
        .size()
        .reset_index(name="positive_ratings")
    )

    candidates = movies[
        ~movies["movieId"].isin(user_seen_movies)
    ].copy()

    candidates = candidates.merge(
        popularity_scores,
        on="movieId",
        how="left",
    )

    candidates["positive_ratings"] = candidates["positive_ratings"].fillna(0)

    def calculate_genre_match_score(genres: str) -> float:
        movie_genres = genres.split("|")

        score = 0.0

        for genre in movie_genres:
            score += genre_weights.get(genre, 0.0)

        return score / len(movie_genres)

    candidates["genre_match_score"] = candidates["genres"].apply(
        calculate_genre_match_score
    )

    max_genre_score = candidates["genre_match_score"].max()

    if max_genre_score > 0:
        candidates["genre_match_score"] = (
            candidates["genre_match_score"] / max_genre_score
        )

    candidates["popularity_score"] = candidates["positive_ratings"].apply(
        math.log1p
    )

    max_popularity_score = candidates["popularity_score"].max()

    if max_popularity_score > 0:
        candidates["popularity_score"] = (
            candidates["popularity_score"] / max_popularity_score
        )

    candidates["hybrid_score"] = (
        genre_weight * candidates["genre_match_score"]
        + popularity_weight * candidates["popularity_score"]
    )

    recommendations = candidates.sort_values(
        ["hybrid_score", "positive_ratings", "movieId"],
        ascending=[False, False, True],
    )

    return recommendations[
        [
            "movieId",
            "title",
            "genres",
            "genre_match_score",
            "popularity_score",
            "hybrid_score",
            "positive_ratings",
        ]
    ].head(k).reset_index(drop=True)


def recommend_genre_boosted_popular_movies(
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    k: int = 10,
    popularity_weight: float = 0.7,
    genre_weight: float = 0.3,
    min_positive_ratings: int = 10,
) -> pd.DataFrame:
    """Recommend popular unseen movies with a smaller user-genre boost."""
    user_seen_movies = train_ratings.loc[
        train_ratings["userId"] == user_id,
        "movieId",
    ].unique()

    user_positive_ratings = train_ratings[
        (train_ratings["userId"] == user_id)
        & (train_ratings["is_positive"])
    ]

    if user_positive_ratings.empty:
        return recommend_popular_movies(
            train_ratings=train_ratings,
            movies=movies,
            user_id=user_id,
            k=k,
        )

    user_positive_movies = user_positive_ratings.merge(
        movies,
        on="movieId",
        how="left",
    )

    genre_weights = {}

    for genres in user_positive_movies["genres"].dropna():
        movie_genres = genres.split("|")
        genre_credit = 1 / len(movie_genres)

        for genre in movie_genres:
            genre_weights[genre] = genre_weights.get(genre, 0) + genre_credit

    total_genre_weight = sum(genre_weights.values())

    for genre in genre_weights:
        genre_weights[genre] = genre_weights[genre] / total_genre_weight

    positive_ratings = train_ratings[train_ratings["is_positive"]]

    popularity_scores = (
        positive_ratings.groupby("movieId")
        .size()
        .reset_index(name="positive_ratings")
    )

    candidates = movies[
        ~movies["movieId"].isin(user_seen_movies)
    ].copy()

    candidates = candidates.merge(
        popularity_scores,
        on="movieId",
        how="left",
    )

    candidates["positive_ratings"] = candidates["positive_ratings"].fillna(0)

    evidence_candidates = candidates[
        candidates["positive_ratings"] >= min_positive_ratings
    ].copy()

    if len(evidence_candidates) >= k:
        candidates = evidence_candidates

    def calculate_genre_match_score(genres: str) -> float:
        movie_genres = genres.split("|")
        score = 0.0

        for genre in movie_genres:
            score += genre_weights.get(genre, 0.0)

        return score / len(movie_genres)

    candidates["genre_match_score"] = candidates["genres"].apply(
        calculate_genre_match_score
    )

    max_genre_score = candidates["genre_match_score"].max()

    if max_genre_score > 0:
        candidates["genre_match_score"] = (
            candidates["genre_match_score"] / max_genre_score
        )

    candidates["popularity_score"] = candidates["positive_ratings"].apply(
        math.log1p
    )

    max_popularity_score = candidates["popularity_score"].max()

    if max_popularity_score > 0:
        candidates["popularity_score"] = (
            candidates["popularity_score"] / max_popularity_score
        )

    candidates["boosted_score"] = (
        popularity_weight * candidates["popularity_score"]
        + genre_weight * candidates["genre_match_score"]
    )

    recommendations = candidates.sort_values(
        ["boosted_score", "positive_ratings", "movieId"],
        ascending=[False, False, True],
    )

    return recommendations[
        [
            "movieId",
            "title",
            "genres",
            "popularity_score",
            "genre_match_score",
            "boosted_score",
            "positive_ratings",
        ]
    ].head(k).reset_index(drop=True)


def recommend_content_based_movies(
    train_ratings: pd.DataFrame,
    movie_content: pd.DataFrame,
    user_id: int,
    k: int = 10,
) -> pd.DataFrame:
    """Recommend unseen movies based on similarity to a user's liked movie content."""
    user_seen_movies = train_ratings.loc[
        train_ratings["userId"] == user_id,
        "movieId",
    ].unique()

    user_positive_ratings = train_ratings[
        (train_ratings["userId"] == user_id)
        & (train_ratings["is_positive"])
    ]

    if user_positive_ratings.empty:
        return pd.DataFrame(
            columns=["movieId", "title", "genres", "content_similarity"]
        )

    _, content_vectors = build_content_tfidf_matrix(movie_content)

    return recommend_content_based_movies_from_vectors(
        train_ratings=train_ratings,
        movie_content=movie_content,
        content_vectors=content_vectors,
        user_id=user_id,
        k=k,
    )


def build_content_tfidf_matrix(
    movie_content: pd.DataFrame,
    max_features: int = 5000,
    text_column: str = "content_text",
):
    """Build reusable TF-IDF features from movie content text."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    content_vectors = vectorizer.fit_transform(movie_content[text_column].fillna(""))

    return vectorizer, content_vectors


def recommend_content_based_movies_from_vectors(
    train_ratings: pd.DataFrame,
    movie_content: pd.DataFrame,
    content_vectors,
    user_id: int,
    k: int = 10,
) -> pd.DataFrame:
    """Recommend content-similar unseen movies using precomputed TF-IDF vectors."""
    user_seen_movies = train_ratings.loc[
        train_ratings["userId"] == user_id,
        "movieId",
    ].unique()

    user_positive_ratings = train_ratings[
        (train_ratings["userId"] == user_id)
        & (train_ratings["is_positive"])
    ]

    if user_positive_ratings.empty:
        return pd.DataFrame(
            columns=["movieId", "title", "genres", "content_similarity"]
        )

    liked_movie_ids = set(user_positive_ratings["movieId"])

    liked_movie_indices = movie_content[
        movie_content["movieId"].isin(liked_movie_ids)
    ].index

    if len(liked_movie_indices) == 0:
        return pd.DataFrame(
            columns=["movieId", "title", "genres", "content_similarity"]
        )

    user_profile = content_vectors[liked_movie_indices].mean(axis=0)
    user_profile = np.asarray(user_profile)

    similarity_scores = cosine_similarity(
        user_profile,
        content_vectors,
    ).flatten()

    candidates = movie_content[
        ~movie_content["movieId"].isin(user_seen_movies)
    ].copy()

    candidates["content_similarity"] = similarity_scores[candidates.index]

    recommendations = candidates.sort_values(
        ["content_similarity", "movieId"],
        ascending=[False, True],
    )

    return recommendations[
        ["movieId", "title", "genres", "content_similarity"]
    ].head(k).reset_index(drop=True)


def recommend_true_hybrid_movies_from_vectors(
    train_ratings: pd.DataFrame,
    movie_content: pd.DataFrame,
    content_vectors,
    user_id: int,
    k: int = 10,
    popularity_weight: float = 0.55,
    content_weight: float = 0.30,
    genre_weight: float = 0.15,
    min_positive_ratings: int = 10,
) -> pd.DataFrame:
    """Recommend unseen movies using popularity, content similarity, and genre taste."""
    total_weight = popularity_weight + content_weight + genre_weight

    if total_weight <= 0:
        raise ValueError("Hybrid weights must sum to a positive value.")

    popularity_weight = popularity_weight / total_weight
    content_weight = content_weight / total_weight
    genre_weight = genre_weight / total_weight

    user_seen_movies = train_ratings.loc[
        train_ratings["userId"] == user_id,
        "movieId",
    ].unique()

    user_positive_ratings = train_ratings[
        (train_ratings["userId"] == user_id)
        & (train_ratings["is_positive"])
    ]

    if user_positive_ratings.empty:
        return recommend_popular_movies(
            train_ratings=train_ratings,
            movies=movie_content,
            user_id=user_id,
            k=k,
        )

    movie_content = movie_content.reset_index(drop=True).copy()

    positive_ratings = train_ratings[train_ratings["is_positive"]]

    popularity_scores = (
        positive_ratings.groupby("movieId")
        .size()
        .reset_index(name="positive_ratings")
    )

    candidates = movie_content[
        ~movie_content["movieId"].isin(user_seen_movies)
    ].copy()

    candidates = candidates.merge(
        popularity_scores,
        on="movieId",
        how="left",
    )

    candidates["positive_ratings"] = candidates["positive_ratings"].fillna(0)

    evidence_candidates = candidates[
        candidates["positive_ratings"] >= min_positive_ratings
    ].copy()

    if len(evidence_candidates) >= k:
        candidates = evidence_candidates

    candidates["popularity_score"] = candidates["positive_ratings"].apply(
        math.log1p
    )

    max_popularity_score = candidates["popularity_score"].max()

    if max_popularity_score > 0:
        candidates["popularity_score"] = (
            candidates["popularity_score"] / max_popularity_score
        )

    liked_movie_ids = set(user_positive_ratings["movieId"])

    liked_movie_indices = movie_content[
        movie_content["movieId"].isin(liked_movie_ids)
    ].index

    if len(liked_movie_indices) == 0:
        candidates["content_similarity"] = 0.0
    else:
        user_profile = content_vectors[liked_movie_indices].mean(axis=0)
        user_profile = np.asarray(user_profile)

        similarity_scores = cosine_similarity(
            user_profile,
            content_vectors,
        ).flatten()

        candidates["content_similarity"] = similarity_scores[candidates.index]

        max_content_score = candidates["content_similarity"].max()

        if max_content_score > 0:
            candidates["content_similarity"] = (
                candidates["content_similarity"] / max_content_score
            )

    user_positive_movies = user_positive_ratings.merge(
        movie_content[["movieId", "genres"]],
        on="movieId",
        how="left",
    )

    genre_weights = {}

    for genres in user_positive_movies["genres"].dropna():
        movie_genres = genres.split("|")
        genre_credit = 1 / len(movie_genres)

        for genre in movie_genres:
            genre_weights[genre] = genre_weights.get(genre, 0) + genre_credit

    total_genre_weight = sum(genre_weights.values())

    if total_genre_weight > 0:
        for genre in genre_weights:
            genre_weights[genre] = genre_weights[genre] / total_genre_weight

    def calculate_genre_match_score(genres: str) -> float:
        movie_genres = genres.split("|")
        score = 0.0

        for genre in movie_genres:
            score += genre_weights.get(genre, 0.0)

        return score / len(movie_genres)

    candidates["genre_match_score"] = candidates["genres"].apply(
        calculate_genre_match_score
    )

    max_genre_score = candidates["genre_match_score"].max()

    if max_genre_score > 0:
        candidates["genre_match_score"] = (
            candidates["genre_match_score"] / max_genre_score
        )

    candidates["hybrid_score"] = (
        popularity_weight * candidates["popularity_score"]
        + content_weight * candidates["content_similarity"]
        + genre_weight * candidates["genre_match_score"]
    )

    recommendations = candidates.sort_values(
        ["hybrid_score", "positive_ratings", "movieId"],
        ascending=[False, False, True],
    )

    return recommendations[
        [
            "movieId",
            "title",
            "genres",
            "popularity_score",
            "content_similarity",
            "genre_match_score",
            "hybrid_score",
            "positive_ratings",
        ]
    ].head(k).reset_index(drop=True)


def recommend_popularity_content_hybrid_movies_from_vectors(
    train_ratings: pd.DataFrame,
    movie_content: pd.DataFrame,
    content_vectors,
    user_id: int,
    k: int = 10,
    popularity_weight: float = 0.80,
    content_weight: float = 0.20,
    min_positive_ratings: int = 10,
) -> pd.DataFrame:
    """Recommend unseen movies using popularity and semantic content similarity."""
    total_weight = popularity_weight + content_weight

    if total_weight <= 0:
        raise ValueError("Hybrid weights must sum to a positive value.")

    popularity_weight = popularity_weight / total_weight
    content_weight = content_weight / total_weight

    user_seen_movies = train_ratings.loc[
        train_ratings["userId"] == user_id,
        "movieId",
    ].unique()

    user_positive_ratings = train_ratings[
        (train_ratings["userId"] == user_id)
        & (train_ratings["is_positive"])
    ]

    if user_positive_ratings.empty:
        return recommend_popular_movies(
            train_ratings=train_ratings,
            movies=movie_content,
            user_id=user_id,
            k=k,
        )

    movie_content = movie_content.reset_index(drop=True).copy()

    positive_ratings = train_ratings[train_ratings["is_positive"]]

    popularity_scores = (
        positive_ratings.groupby("movieId")
        .size()
        .reset_index(name="positive_ratings")
    )

    candidates = movie_content[
        ~movie_content["movieId"].isin(user_seen_movies)
    ].copy()

    candidates = candidates.merge(
        popularity_scores,
        on="movieId",
        how="left",
    )

    candidates["positive_ratings"] = candidates["positive_ratings"].fillna(0)

    evidence_candidates = candidates[
        candidates["positive_ratings"] >= min_positive_ratings
    ].copy()

    if len(evidence_candidates) >= k:
        candidates = evidence_candidates

    candidates["popularity_score"] = candidates["positive_ratings"].apply(
        math.log1p
    )

    max_popularity_score = candidates["popularity_score"].max()

    if max_popularity_score > 0:
        candidates["popularity_score"] = (
            candidates["popularity_score"] / max_popularity_score
        )

    liked_movie_ids = set(user_positive_ratings["movieId"])

    liked_movie_indices = movie_content[
        movie_content["movieId"].isin(liked_movie_ids)
    ].index

    if len(liked_movie_indices) == 0:
        candidates["content_similarity"] = 0.0
    else:
        user_profile = content_vectors[liked_movie_indices].mean(axis=0)
        user_profile = np.asarray(user_profile)

        similarity_scores = cosine_similarity(
            user_profile,
            content_vectors,
        ).flatten()

        candidates["content_similarity"] = similarity_scores[candidates.index]

        max_content_score = candidates["content_similarity"].max()

        if max_content_score > 0:
            candidates["content_similarity"] = (
                candidates["content_similarity"] / max_content_score
            )

    candidates["hybrid_score"] = (
        popularity_weight * candidates["popularity_score"]
        + content_weight * candidates["content_similarity"]
    )

    recommendations = candidates.sort_values(
        ["hybrid_score", "positive_ratings", "movieId"],
        ascending=[False, False, True],
    )

    return recommendations[
        [
            "movieId",
            "title",
            "genres",
            "popularity_score",
            "content_similarity",
            "hybrid_score",
            "positive_ratings",
        ]
    ].head(k).reset_index(drop=True)
