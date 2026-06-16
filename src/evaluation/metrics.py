"""
Evaluation metrics — both point-wise (RMSE, MAE) and ranking (NDCG, Precision, Recall).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute point-wise regression metrics."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": rmse, "mae": mae}


def compute_ranking_metrics(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    k: int = 10,
    rating_threshold: float = 0.6,  # normalized threshold ≈ rating 3.5/5
) -> Dict[str, float]:
    """Compute ranking metrics: NDCG@k, Precision@k, Recall@k.

    Groups test interactions by user and computes per-user ranking quality.
    """
    ndcg_scores, precision_scores, recall_scores = [], [], []
    all_items = test_df["movie_idx"].unique()

    for user_id, user_data in test_df.groupby("user_idx"):
        if len(user_data) < 2:
            continue

        # Score all items in the test set for this user
        user_ids = np.full(len(all_items), user_id, dtype=np.int32)
        item_ids = all_items.astype(np.int32)

        preds = model.predict(
            (user_ids, item_ids), batch_size=512, verbose=0
        ).flatten()

        # Rank by predicted score descending
        ranked_items = all_items[np.argsort(-preds)][:k]

        # Ground truth: items rated above threshold
        relevant = set(
            user_data.loc[user_data["rating_normalized"] >= rating_threshold, "movie_idx"]
        )
        if not relevant:
            continue

        recommended = list(ranked_items)
        hits = [1 if item in relevant else 0 for item in recommended]

        # NDCG@k
        ndcg = _ndcg_at_k(hits, k)
        ndcg_scores.append(ndcg)

        # Precision@k
        precision = sum(hits) / k
        precision_scores.append(precision)

        # Recall@k
        recall = sum(hits) / len(relevant) if relevant else 0.0
        recall_scores.append(recall)

    results = {
        f"ndcg_at_{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        f"precision_at_{k}": float(np.mean(precision_scores)) if precision_scores else 0.0,
        f"recall_at_{k}": float(np.mean(recall_scores)) if recall_scores else 0.0,
    }
    logger.info(f"Ranking metrics @{k}: {results}")
    return results


def _dcg_at_k(hits: list[int], k: int) -> float:
    """Discounted Cumulative Gain."""
    return sum(h / np.log2(i + 2) for i, h in enumerate(hits[:k]))


def _ndcg_at_k(hits: list[int], k: int) -> float:
    """Normalized DCG: DCG / IDCG."""
    dcg = _dcg_at_k(hits, k)
    ideal_hits = sorted(hits, reverse=True)
    idcg = _dcg_at_k(ideal_hits, k)
    return dcg / idcg if idcg > 0 else 0.0
