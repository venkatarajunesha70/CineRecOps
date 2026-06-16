"""
Preprocessing pipeline — cleans, filters, splits, and encodes the dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    """End-to-end preprocessing for the MovieLens recommendation dataset."""

    def __init__(
        self,
        min_ratings_per_user: int = 5,
        min_ratings_per_movie: int = 10,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        self.min_ratings_per_user = min_ratings_per_user
        self.min_ratings_per_movie = min_ratings_per_movie
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed

        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        users: pd.DataFrame | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter, encode, and split into train/val/test sets.

        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames.
        """
        df = self._merge_data(ratings, movies, users)
        df = self._filter_data(df)
        df = self._encode_ids(df, fit=True)
        df = self._normalize_ratings(df)

        train, val, test = self._split(df)
        self._fitted = True

        logger.info(
            f"Split sizes — train: {len(train):,} | val: {len(val):,} | test: {len(test):,}"
        )
        return train, val, test

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders to new data (inference time)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        return self._encode_ids(df, fit=False)

    def save(self, output_dir: str | Path) -> None:
        """Persist encoder mappings for later use in inference."""
        import pickle

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "user_encoder.pkl", "wb") as f:
            pickle.dump(self.user_encoder, f)
        with open(out / "movie_encoder.pkl", "wb") as f:
            pickle.dump(self.movie_encoder, f)

        logger.info(f"Preprocessor artifacts saved to {out}.")

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "DataPreprocessor":
        """Load a persisted preprocessor from disk."""
        import pickle

        artifact_dir = Path(artifact_dir)
        instance = cls()
        with open(artifact_dir / "user_encoder.pkl", "rb") as f:
            instance.user_encoder = pickle.load(f)
        with open(artifact_dir / "movie_encoder.pkl", "rb") as f:
            instance.movie_encoder = pickle.load(f)
        instance._fitted = True
        return instance

    @property
    def n_users(self) -> int:
        return len(self.user_encoder.classes_)

    @property
    def n_movies(self) -> int:
        return len(self.movie_encoder.classes_)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _merge_data(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        users: pd.DataFrame | None,
    ) -> pd.DataFrame:
        df = ratings.merge(movies[["movie_id", "title_clean", "genres", "year"]], on="movie_id", how="left")
        if users is not None:
            df = df.merge(users[["user_id", "gender", "age", "occupation"]], on="user_id", how="left")
        return df

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        original_size = len(df)

        # Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
        df = df[df["user_id"].isin(valid_users)]

        # Filter movies
        movie_counts = df["movie_id"].value_counts()
        valid_movies = movie_counts[movie_counts >= self.min_ratings_per_movie].index
        df = df[df["movie_id"].isin(valid_movies)]

        logger.info(
            f"Filtered: {original_size:,} → {len(df):,} ratings "
            f"({df['user_id'].nunique():,} users, {df['movie_id'].nunique():,} movies)"
        )
        return df.reset_index(drop=True)

    def _encode_ids(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        if fit:
            df["user_idx"] = self.user_encoder.fit_transform(df["user_id"])
            df["movie_idx"] = self.movie_encoder.fit_transform(df["movie_id"])
        else:
            df["user_idx"] = self.user_encoder.transform(df["user_id"])
            df["movie_idx"] = self.movie_encoder.transform(df["movie_id"])
        return df

    def _normalize_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize ratings to [0, 1] range."""
        df = df.copy()
        df["rating_normalized"] = (df["rating"] - 1.0) / 4.0  # MovieLens: 1–5
        return df

    def _split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal split — ensures no future data leaks into training."""
        df = df.sort_values("timestamp").reset_index(drop=True)

        n = len(df)
        test_idx = int(n * (1 - self.test_size))
        val_idx = int(test_idx * (1 - self.val_size))

        train = df.iloc[:val_idx]
        val = df.iloc[val_idx:test_idx]
        test = df.iloc[test_idx:]
        return train, val, test
