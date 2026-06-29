import pandas as pd

from src.content import build_movie_content_features


def test_build_movie_content_features_creates_content_text():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Movie Without Metadata (2000)"],
            "genres": ["Adventure|Animation|Children|Comedy|Fantasy", "Drama"],
        }
    )

    tmdb_metadata = pd.DataFrame(
        {
            "movieId": [1],
            "tmdb_title": ["Toy Story"],
            "overview": ["A cowboy doll feels threatened by a new space toy."],
            "tagline": ["The adventure takes off!"],
            "tmdb_genres": ["Animation, Comedy, Family"],
        }
    )

    content = build_movie_content_features(movies, tmdb_metadata)

    assert len(content) == 2
    assert "content_text" in content.columns
    assert "overview_text" in content.columns
    assert "overview_tagline_text" in content.columns
    assert "overview_genre_text" in content.columns

    toy_story_text = content.loc[content["movieId"] == 1, "content_text"].iloc[0]
    missing_metadata_text = content.loc[content["movieId"] == 2, "content_text"].iloc[0]
    toy_story_overview_genre_text = content.loc[
        content["movieId"] == 1,
        "overview_genre_text",
    ].iloc[0]

    assert "Toy Story" in toy_story_text
    assert "space toy" in toy_story_text
    assert "Adventure Animation Children Comedy Fantasy" in toy_story_text
    assert "Animation  Comedy  Family" in toy_story_text
    assert "Toy Story" not in toy_story_overview_genre_text
    assert "space toy" in toy_story_overview_genre_text

    assert "Movie Without Metadata" in missing_metadata_text
    assert "Drama" in missing_metadata_text
