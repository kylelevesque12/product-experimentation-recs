from pathlib import Path
import time

import pandas as pd
import requests
from tqdm import tqdm

from src.tmdb import fetch_tmdb_movie_details, parse_tmdb_movie_details


LINKS_PATH = Path("data/raw/ml-latest-small/links.csv")
OUTPUT_PATH = Path("data/processed/tmdb_movie_metadata.csv")


def main():
    links = pd.read_csv(LINKS_PATH)
    links = links.dropna(subset=["tmdbId"]).copy()

    if OUTPUT_PATH.exists():
        existing_metadata = pd.read_csv(OUTPUT_PATH)
        completed_tmdb_ids = set(existing_metadata["tmdbId"].astype(int))
        rows = existing_metadata.to_dict("records")
    else:
        completed_tmdb_ids = set()
        rows = []

    movies_to_fetch = links[~links["tmdbId"].astype(int).isin(completed_tmdb_ids)]

    print(f"Already completed: {len(completed_tmdb_ids)} movies")
    print(f"Remaining to fetch: {len(movies_to_fetch)} movies")

    for index, row in tqdm(movies_to_fetch.iterrows(), total=len(movies_to_fetch)):
        movie_id = int(row["movieId"])
        tmdb_id = int(row["tmdbId"])

        try:
            details = fetch_tmdb_movie_details(tmdb_id)
        except requests.HTTPError as error:
            print(f"Skipping movieId={movie_id}, tmdbId={tmdb_id}: {error}")
            continue

        parsed = parse_tmdb_movie_details(
            movie_id=movie_id,
            tmdb_id=tmdb_id,
            details=details,
        )

        rows.append(parsed)

        if len(rows) % 100 == 0:
            metadata = pd.DataFrame(rows)
            metadata.to_csv(OUTPUT_PATH, index=False)

        time.sleep(0.05)

    metadata = pd.DataFrame(rows)
    metadata.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved metadata to {OUTPUT_PATH}")
    print(f"Total saved movies: {len(metadata)}")


if __name__ == "__main__":
    main()
