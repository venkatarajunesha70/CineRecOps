"""Script to run preprocessing and feature engineering and save artifacts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger

from data.ingestion import load_movies, load_ratings, load_users
from data.preprocessing import DataPreprocessor
from features.feature_engineering import FeatureEngineer


def main():
    raw_dir = Path("data/raw/ml-1m")
    processed_dir = Path("data/processed")
    features_dir = Path("data/features")

    logger.info("Loading raw data ...")
    ratings = load_ratings(raw_dir)
    movies = load_movies(raw_dir)
    users = load_users(raw_dir)

    logger.info("Preprocessing ...")
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.fit_transform(ratings, movies, users)
    preprocessor.save(processed_dir)

    # Save splits
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)
    movies.to_parquet(processed_dir / "movies.parquet", index=False)

    logger.info("Feature engineering ...")
    feat_eng = FeatureEngineer()
    user_feats, item_feats = feat_eng.fit_transform(train_df, movies)
    feat_eng.save(features_dir)

    features_dir.mkdir(parents=True, exist_ok=True)
    user_feats.to_parquet(features_dir / "user_features.parquet", index=False)
    item_feats.to_parquet(features_dir / "item_features.parquet", index=False)

    logger.info("All artifacts saved.")
    logger.info(f"  train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")


if __name__ == "__main__":
    main()
