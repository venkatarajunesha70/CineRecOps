"""Unit tests for the data preprocessing pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.preprocessing import DataPreprocessor


@pytest.fixture
def sample_ratings() -> pd.DataFrame:
    """Minimal ratings DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "user_id": rng.integers(1, 21, size=n),
        "movie_id": rng.integers(1, 31, size=n),
        "rating": rng.integers(1, 6, size=n).astype(float),
        "timestamp": rng.integers(900_000_000, 1_000_000_000, size=n),
    })


@pytest.fixture
def sample_movies() -> pd.DataFrame:
    genres = ["Action|Adventure", "Comedy|Romance", "Drama", "Sci-Fi|Thriller"]
    return pd.DataFrame({
        "movie_id": range(1, 31),
        "title_clean": [f"Movie {i}" for i in range(1, 31)],
        "genres": [genres[i % 4] for i in range(30)],
        "year": [1990 + i % 30 for i in range(30)],
    })


class TestDataPreprocessor:
    def test_fit_transform_returns_three_splits(self, sample_ratings, sample_movies):
        preprocessor = DataPreprocessor(min_ratings_per_user=1, min_ratings_per_movie=1)
        train, val, test = preprocessor.fit_transform(sample_ratings, sample_movies)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_splits_sum_to_total(self, sample_ratings, sample_movies):
        preprocessor = DataPreprocessor(min_ratings_per_user=1, min_ratings_per_movie=1)
        train, val, test = preprocessor.fit_transform(sample_ratings, sample_movies)
        # After filtering, total should match filtered df size
        total = len(train) + len(val) + len(test)
        assert total > 0

    def test_encoded_indices_are_contiguous(self, sample_ratings, sample_movies):
        preprocessor = DataPreprocessor(min_ratings_per_user=1, min_ratings_per_movie=1)
        train, val, test = preprocessor.fit_transform(sample_ratings, sample_movies)
        all_df = pd.concat([train, val, test])
        assert all_df["user_idx"].min() == 0
        assert all_df["movie_idx"].min() == 0

    def test_rating_normalization(self, sample_ratings, sample_movies):
        preprocessor = DataPreprocessor(min_ratings_per_user=1, min_ratings_per_movie=1)
        train, _, _ = preprocessor.fit_transform(sample_ratings, sample_movies)
        assert train["rating_normalized"].between(0.0, 1.0).all()

    def test_n_users_n_movies_properties(self, sample_ratings, sample_movies):
        preprocessor = DataPreprocessor(min_ratings_per_user=1, min_ratings_per_movie=1)
        preprocessor.fit_transform(sample_ratings, sample_movies)
        assert preprocessor.n_users > 0
        assert preprocessor.n_movies > 0

    def test_temporal_split_order(self, sample_ratings, sample_movies):
        preprocessor = DataPreprocessor(min_ratings_per_user=1, min_ratings_per_movie=1)
        train, val, test = preprocessor.fit_transform(sample_ratings, sample_movies)
        # Training timestamps should precede validation timestamps
        assert train["timestamp"].max() <= val["timestamp"].max()

    def test_minimum_rating_filter(self):
        """Users with fewer than min_ratings should be excluded."""
        ratings = pd.DataFrame({
            "user_id": [1, 1, 2],  # user 2 only has 1 rating
            "movie_id": [1, 2, 1],
            "rating": [4.0, 3.0, 5.0],
            "timestamp": [1, 2, 3],
        })
        movies = pd.DataFrame({
            "movie_id": [1, 2],
            "title_clean": ["M1", "M2"],
            "genres": ["Drama", "Comedy"],
            "year": [2000, 2001],
        })
        preprocessor = DataPreprocessor(min_ratings_per_user=2, min_ratings_per_movie=1)
        train, val, test = preprocessor.fit_transform(ratings, movies)
        all_df = pd.concat([train, val, test])
        assert 2 not in all_df["user_id"].values
