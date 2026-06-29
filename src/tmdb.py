import os
from pathlib import Path
import time

import pandas as pd
import requests
from tqdm import tqdm


TMDB_MOVIE_DETAILS_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"


def fetch_tmdb_movie_details(tmdb_id: int) -> dict:
    """Fetch movie details from TMDb for one TMDb movie ID."""
    api_key = os.environ["TMDB_API_KEY"]

    response = requests.get(
        TMDB_MOVIE_DETAILS_URL.format(tmdb_id=tmdb_id),
        params={"api_key": api_key},
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def parse_tmdb_movie_details(movie_id: int, tmdb_id: int, details: dict) -> dict:
    """Extract the TMDb fields we want to keep."""
    genres = details.get("genres", [])
    genre_names = [genre["name"] for genre in genres]

    return {
        "movieId": movie_id,
        "tmdbId": tmdb_id,
        "tmdb_title": details.get("title"),
        "overview": details.get("overview"),
        "tagline": details.get("tagline"),
        "release_date": details.get("release_date"),
        "runtime": details.get("runtime"),
        "tmdb_popularity": details.get("popularity"),
        "tmdb_vote_average": details.get("vote_average"),
        "tmdb_vote_count": details.get("vote_count"),
        "tmdb_genres": "|".join(genre_names),
    }


def fetch_tmdb_metadata_batch(
    links: pd.DataFrame,
    output_path: Path,
    limit: int | None = None,
    sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    """Fetch TMDb metadata for MovieLens movies and save the results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    links_with_tmdb = links.dropna(subset=["tmdbId"]).copy()

    if limit is not None:
        links_with_tmdb = links_with_tmdb.head(limit)

    for _, row in tqdm(links_with_tmdb.iterrows(), total=len(links_with_tmdb)):
        movie_id = int(row["movieId"])
        tmdb_id = int(row["tmdbId"])

        details = fetch_tmdb_movie_details(tmdb_id)
        parsed = parse_tmdb_movie_details(
            movie_id=movie_id,
            tmdb_id=tmdb_id,
            details=details,
        )

        rows.append(parsed)

        time.sleep(sleep_seconds)

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_path, index=False)

    return metadata