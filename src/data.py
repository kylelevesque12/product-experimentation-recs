from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests


MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_ZIP_NAME = "ml-latest-small.zip"
MOVIELENS_FOLDER_NAME = "ml-latest-small"


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    """Return the main data directory."""
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """Return the directory for raw downloaded data."""
    return get_data_dir() / "raw"


def get_processed_data_dir() -> Path:
    """Return the directory for cleaned data files."""
    return get_data_dir() / "processed"


def download_movielens() -> Path:
    """Download the MovieLens latest-small zip file if needed."""
    raw_data_dir = get_raw_data_dir()
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_data_dir / MOVIELENS_ZIP_NAME

    if zip_path.exists():
        return zip_path

    response = requests.get(MOVIELENS_SMALL_URL, timeout=30)
    response.raise_for_status()

    zip_path.write_bytes(response.content)
    return zip_path


def extract_movielens() -> Path:
    """Extract the MovieLens latest-small zip file if needed."""
    zip_path = download_movielens()
    extracted_dir = get_raw_data_dir() / MOVIELENS_FOLDER_NAME

    if extracted_dir.exists():
        return extracted_dir

    with ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(get_raw_data_dir())
    return extracted_dir


def load_ratings() -> pd.DataFrame:
    """Load the raw MovieLens ratings."""
    data_dir = extract_movielens()
    ratings_path = data_dir / "ratings.csv"
    return pd.read_csv(ratings_path)


def load_movies() -> pd.DataFrame:
    """Load the raw MovieLens movies."""
    data_dir = extract_movielens()
    movies_path = data_dir / "movies.csv"
    return pd.read_csv(movies_path)


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw MovieLens ratings for recommendation experiments."""
    cleaned = ratings.copy()

    cleaned["rated_at"] = pd.to_datetime(cleaned["timestamp"], unit="s")
    cleaned["is_positive"] = cleaned["rating"] >= 4.0

    return cleaned


def temporal_train_test_split(
    ratings: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user's ratings into earlier train rows and later test rows."""
    if test_fraction <= 0 or test_fraction >= 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    sorted_ratings = ratings.sort_values(["userId", "rated_at"]).copy()

    train_parts = []
    test_parts = []

    for _, user_ratings in sorted_ratings.groupby("userId"):
        n_ratings = len(user_ratings)

        if n_ratings < 2:
            train_parts.append(user_ratings)
            continue

        n_test = max(1, int(n_ratings * test_fraction))
        split_index = n_ratings - n_test

        train_parts.append(user_ratings.iloc[:split_index])
        test_parts.append(user_ratings.iloc[split_index:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)

    return train, test