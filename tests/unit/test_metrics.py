"""Unit tests for evaluation metrics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from evaluation.metrics import _dcg_at_k, _ndcg_at_k, compute_metrics


class TestPointwiseMetrics:
    def test_rmse_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y, y)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)

    def test_mae_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y, y)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)

    def test_rmse_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # all off by 1
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(1.0, abs=1e-6)

    def test_mae_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(1.0, abs=1e-6)


class TestRankingMetrics:
    def test_dcg_all_relevant(self):
        hits = [1, 1, 1, 1, 1]
        dcg = _dcg_at_k(hits, k=5)
        assert dcg > 0

    def test_dcg_no_relevant(self):
        hits = [0, 0, 0]
        dcg = _dcg_at_k(hits, k=3)
        assert dcg == pytest.approx(0.0)

    def test_ndcg_perfect_ranking(self):
        hits = [1, 1, 1, 0, 0]
        ndcg = _ndcg_at_k(hits, k=5)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_no_relevant(self):
        hits = [0, 0, 0]
        ndcg = _ndcg_at_k(hits, k=3)
        assert ndcg == pytest.approx(0.0)

    def test_ndcg_partial_relevance(self):
        hits = [1, 0, 1, 0, 0]
        ndcg = _ndcg_at_k(hits, k=5)
        # DCG with relevant items at positions 1,3 is less than IDCG
        assert 0.0 < ndcg < 1.0

    def test_ndcg_first_position_better_than_last(self):
        hits_first = [1, 0, 0, 0, 0]
        hits_last = [0, 0, 0, 0, 1]
        ndcg_first = _ndcg_at_k(hits_first, k=5)
        ndcg_last = _ndcg_at_k(hits_last, k=5)
        assert ndcg_first > ndcg_last
