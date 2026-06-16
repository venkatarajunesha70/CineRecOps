"""Script to download and validate the MovieLens dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.ingestion import download_movielens, load_movies, load_ratings, load_users
from loguru import logger


def main():
    logger.info("Downloading MovieLens-1M dataset ...")
    data_dir = download_movielens(dataset="1m", raw_dir="data/raw")

    ratings = load_ratings(data_dir, dataset="1m")
    movies = load_movies(data_dir, dataset="1m")
    users = load_users(data_dir, dataset="1m")

    logger.info(f"Ratings: {len(ratings):,} rows")
    logger.info(f"Movies:  {len(movies):,} rows")
    if users is not None:
        logger.info(f"Users:   {len(users):,} rows")
    logger.info("Download complete.")


if __name__ == "__main__":
    main()
