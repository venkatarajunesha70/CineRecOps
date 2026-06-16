"""Unit tests for TensorFlow model architectures."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import tensorflow as tf
from models import NeuralCollaborativeFiltering, TwoTowerModel
from models.hybrid import HybridRecommender


N_USERS = 100
N_ITEMS = 200
BATCH = 32


@pytest.fixture
def user_batch():
    return np.random.randint(0, N_USERS, size=BATCH).astype(np.int32)


@pytest.fixture
def item_batch():
    return np.random.randint(0, N_ITEMS, size=BATCH).astype(np.int32)


class TestTwoTowerModel:
    def test_output_shape(self, user_batch, item_batch):
        model = TwoTowerModel(n_users=N_USERS, n_items=N_ITEMS, embedding_dim=16,
                               hidden_layers=[32, 16])
        out = model((user_batch, item_batch), training=False)
        assert out.shape == (BATCH, 1)

    def test_output_range(self, user_batch, item_batch):
        model = TwoTowerModel(n_users=N_USERS, n_items=N_ITEMS, embedding_dim=16,
                               hidden_layers=[32, 16])
        out = model((user_batch, item_batch), training=False).numpy()
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_user_embedding_shape(self, user_batch):
        model = TwoTowerModel(n_users=N_USERS, n_items=N_ITEMS, embedding_dim=32,
                               hidden_layers=[64])
        emb = model.get_user_embedding(user_batch)
        assert emb.shape == (BATCH, 32)

    def test_item_embedding_shape(self, item_batch):
        model = TwoTowerModel(n_users=N_USERS, n_items=N_ITEMS, embedding_dim=32,
                               hidden_layers=[64])
        emb = model.get_item_embedding(item_batch)
        assert emb.shape == (BATCH, 32)

    def test_user_embeddings_are_l2_normalized(self, user_batch):
        model = TwoTowerModel(n_users=N_USERS, n_items=N_ITEMS, embedding_dim=16)
        emb = model.get_user_embedding(user_batch).numpy()
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_compile_and_single_step(self, user_batch, item_batch):
        model = TwoTowerModel(n_users=N_USERS, n_items=N_ITEMS, embedding_dim=16,
                               hidden_layers=[32])
        model.compile(optimizer="adam", loss="mse")
        labels = np.random.rand(BATCH, 1).astype(np.float32)
        loss = model.train_on_batch((user_batch, item_batch), labels)
        assert isinstance(float(loss), float)


class TestNeuralCollaborativeFiltering:
    def test_output_shape(self, user_batch, item_batch):
        model = NeuralCollaborativeFiltering(
            n_users=N_USERS, n_items=N_ITEMS, embedding_dim=16, hidden_layers=[32, 16]
        )
        out = model((user_batch, item_batch), training=False)
        assert out.shape == (BATCH, 1)

    def test_output_range(self, user_batch, item_batch):
        model = NeuralCollaborativeFiltering(
            n_users=N_USERS, n_items=N_ITEMS, embedding_dim=16, hidden_layers=[32, 16]
        )
        out = model((user_batch, item_batch)).numpy()
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_compile_and_single_step(self, user_batch, item_batch):
        model = NeuralCollaborativeFiltering(
            n_users=N_USERS, n_items=N_ITEMS, embedding_dim=16, hidden_layers=[32]
        )
        model.compile(optimizer="adam", loss="mse")
        labels = np.random.rand(BATCH, 1).astype(np.float32)
        loss = model.train_on_batch((user_batch, item_batch), labels)
        assert isinstance(float(loss), float)


class TestHybridRecommender:
    def test_output_shape(self, user_batch, item_batch):
        model = HybridRecommender(
            n_users=N_USERS, n_items=N_ITEMS,
            n_user_features=10, n_item_features=15,
            embedding_dim=16, hidden_layers=[32, 16],
        )
        user_feats = np.random.rand(BATCH, 10).astype(np.float32)
        item_feats = np.random.rand(BATCH, 15).astype(np.float32)
        out = model((user_batch, item_batch, user_feats, item_feats), training=False)
        assert out.shape == (BATCH, 1)
