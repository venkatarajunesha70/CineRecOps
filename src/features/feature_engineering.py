"""
Feature engineering pipeline — builds user and item feature matrices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


# All MovieLens genres
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]


class FeatureEngineer:
    """Builds rich user and item feature representations."""

    def __init__(self) -> None:
        self.genre_binarizer = MultiLabelBinarizer(classes=ALL_GENRES)
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        movies_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build user and item feature DataFrames from training data.

        Returns:
            (user_features_df, item_features_df)
        """
        item_features = self._build_item_features(movies_df, fit=True)
        user_features = self._build_user_features(train_df, movies_df, fit=True)
        self._fitted = True

        logger.info(
            f"Built {user_features.shape[1]} user features and "
            f"{item_features.shape[1]} item features."
        )
        return user_features, item_features

    def transform_users(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        return self._build_user_features(ratings_df, movies_df, fit=False)

    def transform_items(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        return self._build_item_features(movies_df, fit=False)

    def save(self, output_dir: str | Path) -> None:
        import pickle

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "genre_binarizer.pkl", "wb") as f:
            pickle.dump(self.genre_binarizer, f)
        with open(out / "user_scaler.pkl", "wb") as f:
            pickle.dump(self.user_scaler, f)
        with open(out / "item_scaler.pkl", "wb") as f:
            pickle.dump(self.item_scaler, f)
        logger.info(f"Feature engineering artifacts saved to {out}.")

    # ── Item features ─────────────────────────────────────────────────────────

    def _build_item_features(self, movies_df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = movies_df.copy()

        # One-hot encode genres
        genre_lists = df["genres"].fillna("").str.split("|")
        if fit:
            genre_matrix = self.genre_binarizer.fit_transform(genre_lists)
        else:
            genre_matrix = self.genre_binarizer.transform(genre_lists)
        genre_df = pd.DataFrame(
            genre_matrix,
            columns=[f"genre_{g.lower().replace(' ', '_').replace('-', '_')}" for g in ALL_GENRES],
            index=df.index,
        )

        # Popularity: log-scaled rating count
        df["log_rating_count"] = np.log1p(df.get("rating_count", 1))

        # Decade bucket from year
        df["decade"] = (df["year"].fillna(1990) // 10 * 10).astype(int)
        decade_dummies = pd.get_dummies(df["decade"], prefix="decade")

        item_features = pd.concat(
            [df[["movie_id", "log_rating_count"]].reset_index(drop=True),
             genre_df.reset_index(drop=True),
             decade_dummies.reset_index(drop=True)],
            axis=1,
        )
        return item_features

    # ── User features ─────────────────────────────────────────────────────────

    def _build_user_features(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        fit: bool,
    ) -> pd.DataFrame:
        df = ratings_df.merge(movies_df[["movie_id", "genres"]], on="movie_id", how="left")

        # Aggregate rating statistics per user
        agg = (
            df.groupby("user_id")["rating"]
            .agg(
                user_avg_rating="mean",
                user_rating_count="count",
                user_rating_std="std",
            )
            .fillna(0)
            .reset_index()
        )

        # Genre preferences per user: fraction of ratings in each genre
        genre_lists = df["genres"].fillna("").str.split("|")
        genre_matrix = self.genre_binarizer.transform(genre_lists)
        genre_df = pd.DataFrame(genre_matrix, columns=ALL_GENRES)
        genre_df["user_id"] = df["user_id"].values
        genre_pref = genre_df.groupby("user_id")[ALL_GENRES].mean().reset_index()
        genre_pref.columns = ["user_id"] + [
            f"pref_{g.lower().replace(' ', '_').replace('-', '_')}" for g in ALL_GENRES
        ]

        user_features = agg.merge(genre_pref, on="user_id", how="left")

        # Add demographic features if present
        if "gender" in df.columns:
            demo = df[["user_id", "gender", "age", "occupation"]].drop_duplicates("user_id")
            demo["gender_enc"] = (demo["gender"] == "M").astype(int)
            user_features = user_features.merge(
                demo[["user_id", "gender_enc", "age", "occupation"]], on="user_id", how="left"
            )

        # Scale numeric features
        numeric_cols = ["user_avg_rating", "user_rating_count", "user_rating_std"]
        if fit:
            user_features[numeric_cols] = self.user_scaler.fit_transform(
                user_features[numeric_cols].fillna(0)
            )
        else:
            user_features[numeric_cols] = self.user_scaler.transform(
                user_features[numeric_cols].fillna(0)
            )

        return user_features
