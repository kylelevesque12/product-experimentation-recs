import pandas as pd


def build_movie_content_features(movies: pd.DataFrame, tmdb_metadata: pd.DataFrame) -> pd.DataFrame:
    movie_content = movies.merge(tmdb_metadata, on="movieId", how="left")

    movie_content["genres"] = movie_content["genres"].fillna("")
    movie_content["tmdb_genres"] = movie_content["tmdb_genres"].fillna("")
    movie_content["overview"] = movie_content["overview"].fillna("")
    movie_content["tagline"] = movie_content["tagline"].fillna("")

    movie_content["movielens_genre_text"] = movie_content["genres"].str.replace(
        "|",
        " ",
        regex=False,
    )

    movie_content["tmdb_genre_text"] = movie_content["tmdb_genres"].str.replace(
        ",",
        " ",
        regex=False,
    )

    movie_content["overview_text"] = movie_content["overview"]

    movie_content["overview_tagline_text"] = (
        movie_content["overview"]
        + " "
        + movie_content["tagline"]
    )

    movie_content["overview_genre_text"] = (
        movie_content["overview"]
        + " "
        + movie_content["movielens_genre_text"]
        + " "
        + movie_content["tmdb_genre_text"]
    )

    movie_content["content_text"] = (
        movie_content["title"]
        + " "
        + movie_content["movielens_genre_text"]
        + " "
        + movie_content["tmdb_genre_text"]
        + " "
        + movie_content["overview"]
        + " "
        + movie_content["tagline"]
    )

    return movie_content[
        [
            "movieId",
            "title",
            "genres",
            "tmdb_title",
            "overview",
            "tagline",
            "tmdb_genres",
            "overview_text",
            "overview_tagline_text",
            "overview_genre_text",
            "content_text",
        ]
    ]
