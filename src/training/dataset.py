"""
TensorFlow Dataset builders for efficient batched training.
"""
from __future__ import annotations

import pandas as pd
import tensorflow as tf


def build_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = 1024,
    shuffle: bool = True,
    repeat: bool = False,
    buffer_size: int = 10_000,
) -> tf.data.Dataset:
    """Convert a ratings DataFrame into a tf.data.Dataset.

    Args:
        df: Must contain columns: user_idx, movie_idx, rating_normalized.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data each epoch.
        repeat: Whether to repeat indefinitely (for training loops).
        buffer_size: Shuffle buffer size.

    Returns:
        A tf.data.Dataset yielding (inputs_dict, labels) tuples.
    """
    user_idx = df["user_idx"].values.astype("int32")
    item_idx = df["movie_idx"].values.astype("int32")
    labels = df["rating_normalized"].values.astype("float32")

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"user_idx": user_idx, "item_idx": item_idx}, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=42)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_tf_dataset_hybrid(
    df: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    batch_size: int = 1024,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a dataset that includes dense side features for the hybrid model."""
    merged = df.merge(user_features, on="user_id", how="left")
    merged = merged.merge(item_features, on="movie_id", how="left")

    user_feat_cols = [c for c in user_features.columns if c != "user_id"]
    item_feat_cols = [c for c in item_features.columns if c != "movie_id"]

    inputs = {
        "user_idx": merged["user_idx"].values.astype("int32"),
        "item_idx": merged["movie_idx"].values.astype("int32"),
        "user_feats": merged[user_feat_cols].fillna(0).values.astype("float32"),
        "item_feats": merged[item_feat_cols].fillna(0).values.astype("float32"),
    }
    labels = merged["rating_normalized"].values.astype("float32")

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10_000, seed=42)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
