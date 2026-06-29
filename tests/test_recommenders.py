import pandas as pd

from src.recommenders import (
    build_content_tfidf_matrix,
    build_user_genre_profile,
    recommend_content_based_movies,
    recommend_content_based_movies_from_vectors,
    recommend_genre_boosted_popular_movies,
    recommend_popular_movies,
    recommend_popularity_content_hybrid_movies_from_vectors,
    recommend_true_hybrid_movies_from_vectors,
)


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

    
def test_build_user_genre_profile_counts_genres_from_positive_ratings():
    train_ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2],
            "movieId": [10, 20, 30, 40],
            "rating": [5.0, 4.0, 2.0, 5.0],
            "is_positive": [True, True, False, True],
        }
    )

    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": ["Movie A", "Movie B", "Movie C", "Movie D"],
            "genres": [
                "Action|Comedy",
                "Action|Drama",
                "Horror",
                "Comedy",
            ],
        }
    )

    profile = build_user_genre_profile(
        train_ratings=train_ratings,
        movies=movies,
        user_id=1,
    )

    genre_counts = dict(zip(profile["genre"], profile["positive_count"]))

    assert genre_counts["Action"] == 2
    assert genre_counts["Comedy"] == 1
    assert genre_counts["Drama"] == 1
    assert "Horror" not in genre_counts


def test_genre_boosted_popular_movies_uses_popularity_with_genre_boost():
    train_ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3, 4, 5, 6, 7],
            "movieId": [10, 20, 20, 20, 30, 30, 40],
            "rating": [5.0, 5.0, 4.5, 4.0, 5.0, 4.0, 5.0],
            "is_positive": [True, True, True, True, True, True, True],
        }
    )

    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": [
                "Seen Action",
                "Very Popular Comedy",
                "Popular Action",
                "Less Popular Action",
            ],
            "genres": ["Action", "Comedy", "Action", "Action"],
        }
    )

    recommendations = recommend_genre_boosted_popular_movies(
        train_ratings=train_ratings,
        movies=movies,
        user_id=1,
        k=3,
        min_positive_ratings=1,
    )

    assert 10 not in recommendations["movieId"].tolist()
    assert recommendations["movieId"].tolist() == [30, 20, 40]


def test_content_based_recommender_recommends_similar_unseen_movies():
    train_ratings = pd.DataFrame(
        {
            "userId": [1, 1],
            "movieId": [10, 40],
            "rating": [5.0, 2.0],
            "is_positive": [True, False],
        }
    )

    movie_content = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": [
                "Seen Fantasy Movie",
                "Similar Fantasy Movie",
                "Baseball Movie",
                "Seen Horror Movie",
            ],
            "genres": ["Fantasy", "Fantasy", "Sports", "Horror"],
            "content_text": [
                "wizard magic castle dragon quest",
                "wizard magic castle dragon adventure",
                "baseball pitcher stadium playoff team",
                "ghost haunted mansion nightmare",
            ],
        }
    )

    recommendations = recommend_content_based_movies(
        train_ratings=train_ratings,
        movie_content=movie_content,
        user_id=1,
        k=2,
    )

    recommended_movie_ids = recommendations["movieId"].tolist()

    assert 10 not in recommended_movie_ids
    assert 40 not in recommended_movie_ids
    assert recommended_movie_ids[0] == 20
    assert recommendations["content_similarity"].notna().all()


def test_content_based_recommender_can_reuse_tfidf_vectors():
    train_ratings = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
            "rating": [5.0],
            "is_positive": [True],
        }
    )

    movie_content = pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": [
                "Seen Space Movie",
                "Similar Space Movie",
                "Cooking Movie",
            ],
            "genres": ["Sci-Fi", "Sci-Fi", "Comedy"],
            "content_text": [
                "space rocket planet galaxy",
                "space rocket planet mission",
                "kitchen recipe dinner family",
            ],
        }
    )

    _, content_vectors = build_content_tfidf_matrix(movie_content)

    recommendations = recommend_content_based_movies_from_vectors(
        train_ratings=train_ratings,
        movie_content=movie_content,
        content_vectors=content_vectors,
        user_id=1,
        k=2,
    )

    assert recommendations.loc[0, "movieId"] == 20
    assert 10 not in recommendations["movieId"].tolist()


def test_build_content_tfidf_matrix_can_use_selected_text_column():
    movie_content = pd.DataFrame(
        {
            "movieId": [10, 20],
            "title": ["Space Movie", "Cooking Movie"],
            "genres": ["Sci-Fi", "Comedy"],
            "content_text": [
                "space rocket galaxy",
                "kitchen recipe dinner",
            ],
            "overview_text": [
                "astronaut mission",
                "family restaurant",
            ],
        }
    )

    vectorizer, content_vectors = build_content_tfidf_matrix(
        movie_content,
        text_column="overview_text",
    )

    feature_names = set(vectorizer.get_feature_names_out())

    assert content_vectors.shape[0] == 2
    assert "astronaut" in feature_names
    assert "space" not in feature_names


def test_true_hybrid_recommender_combines_content_genre_and_popularity():
    train_ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3, 4, 5, 6, 7],
            "movieId": [10, 20, 30, 30, 30, 30, 30],
            "rating": [5.0, 5.0, 5.0, 4.5, 4.0, 5.0, 4.5],
            "is_positive": [True, True, True, True, True, True, True],
        }
    )

    movie_content = pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": [
                "Seen Space Movie",
                "Similar Space Movie",
                "Popular Cooking Movie",
            ],
            "genres": ["Sci-Fi", "Sci-Fi", "Comedy"],
            "content_text": [
                "space rocket planet galaxy",
                "space rocket planet mission",
                "kitchen recipe dinner family",
            ],
        }
    )

    _, content_vectors = build_content_tfidf_matrix(movie_content)

    recommendations = recommend_true_hybrid_movies_from_vectors(
        train_ratings=train_ratings,
        movie_content=movie_content,
        content_vectors=content_vectors,
        user_id=1,
        k=2,
        popularity_weight=0.35,
        content_weight=0.50,
        genre_weight=0.15,
        min_positive_ratings=1,
    )

    assert recommendations.loc[0, "movieId"] == 20
    assert 10 not in recommendations["movieId"].tolist()
    assert "hybrid_score" in recommendations.columns


def test_popularity_content_hybrid_uses_no_explicit_genre_score():
    train_ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3, 4, 5],
            "movieId": [10, 30, 30, 30, 30],
            "rating": [5.0, 5.0, 4.5, 4.0, 5.0],
            "is_positive": [True, True, True, True, True],
        }
    )

    movie_content = pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": [
                "Seen Space Movie",
                "Similar Space Movie",
                "Popular Cooking Movie",
            ],
            "genres": ["Sci-Fi", "Sci-Fi", "Comedy"],
            "content_text": [
                "space rocket planet galaxy",
                "space rocket planet mission",
                "kitchen recipe dinner family",
            ],
        }
    )

    _, content_vectors = build_content_tfidf_matrix(movie_content)

    recommendations = recommend_popularity_content_hybrid_movies_from_vectors(
        train_ratings=train_ratings,
        movie_content=movie_content,
        content_vectors=content_vectors,
        user_id=1,
        k=2,
        popularity_weight=0.30,
        content_weight=0.70,
        min_positive_ratings=1,
    )

    assert 10 not in recommendations["movieId"].tolist()
    assert "genre_match_score" not in recommendations.columns
    assert "popularity_score" in recommendations.columns
    assert "content_similarity" in recommendations.columns
    assert "hybrid_score" in recommendations.columns
    assert recommendations.loc[
        recommendations["movieId"] == 20,
        "content_similarity",
    ].iloc[0] > recommendations.loc[
        recommendations["movieId"] == 30,
        "content_similarity",
    ].iloc[0]
