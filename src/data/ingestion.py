"""
Data ingestion module — downloads and validates the MovieLens dataset.
"""
import os
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd
from loguru import logger


MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_movielens(
    dataset: str = "1m",
    raw_dir: str = "data/raw",
) -> Path:
    """Download the MovieLens dataset and extract it.

    Args:
        dataset: '1m' for MovieLens-1M or '100k' for MovieLens-100K-small.
        raw_dir: Destination directory for raw data files.

    Returns:
        Path to the extracted directory.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    url = MOVIELENS_1M_URL if dataset == "1m" else MOVIELENS_100K_URL
    zip_name = "ml-1m.zip" if dataset == "1m" else "ml-latest-small.zip"
    zip_path = raw_path / zip_name

    if not zip_path.exists():
        logger.info(f"Downloading MovieLens-{dataset} from {url} ...")
        urllib.request.urlretrieve(url, zip_path, _progress_hook)
        logger.info("Download complete.")
    else:
        logger.info(f"Archive already exists at {zip_path}, skipping download.")

    extract_dir = raw_path / zip_name.replace(".zip", "")
    if not extract_dir.exists():
        logger.info(f"Extracting {zip_path} → {raw_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_path)
        logger.info("Extraction complete.")
    else:
        logger.info(f"Data already extracted at {extract_dir}.")

    return extract_dir


def load_ratings(data_dir: Path, dataset: str = "1m") -> pd.DataFrame:
    """Load ratings file into a DataFrame.

    Supports both MovieLens-1M (.dat separator) and 100K-small (.csv).
    """
    if dataset == "1m":
        ratings_file = data_dir / "ratings.dat"
        df = pd.read_csv(
            ratings_file,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1",
        )
    else:
        ratings_file = data_dir / "ratings.csv"
        df = pd.read_csv(ratings_file)
        df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})

    logger.info(
        f"Loaded {len(df):,} ratings | {df['user_id'].nunique():,} users | "
        f"{df['movie_id'].nunique():,} movies"
    )
    return df


def load_movies(data_dir: Path, dataset: str = "1m") -> pd.DataFrame:
    """Load movies metadata."""
    if dataset == "1m":
        movies_file = data_dir / "movies.dat"
        df = pd.read_csv(
            movies_file,
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
        )
        # Extract year from title: "Toy Story (1995)" → 1995
        df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype("Int64")
        df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()
    else:
        movies_file = data_dir / "movies.csv"
        df = pd.read_csv(movies_file)
        df = df.rename(columns={"movieId": "movie_id"})
        df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype("Int64")
        df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()

    logger.info(f"Loaded {len(df):,} movies.")
    return df


def load_users(data_dir: Path, dataset: str = "1m") -> pd.DataFrame | None:
    """Load user metadata (available in MovieLens-1M only)."""
    if dataset != "1m":
        logger.info("User metadata not available for MovieLens-100K-small.")
        return None

    users_file = data_dir / "users.dat"
    df = pd.read_csv(
        users_file,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    logger.info(f"Loaded {len(df):,} users.")
    return df


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    if block_num % 500 == 0:
        logger.info(f"  ... {pct:.1f}% ({downloaded / 1e6:.1f} MB)")
