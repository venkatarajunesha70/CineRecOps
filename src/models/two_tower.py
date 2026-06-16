"""
Two-Tower (Dual Encoder) Recommendation Model.

Architecture:
  User Tower  → dense user embedding
  Item Tower  → dense item embedding
  Score       → dot product (retrieval) or MLP (ranking)
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


class UserTower(keras.layers.Layer):
    """Encodes user_id + optional side features into a fixed-dim embedding."""

    def __init__(
        self,
        n_users: int,
        embedding_dim: int = 64,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.3,
        l2: float = 0.001,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        hidden_layers = hidden_layers or [256, 128]

        self.user_embedding = keras.layers.Embedding(
            input_dim=n_users + 1,
            output_dim=embedding_dim,
            embeddings_regularizer=keras.regularizers.l2(l2),
            name="user_embedding",
        )
        self.dense_layers = [
            keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2),
            )
            for units in hidden_layers
        ]
        self.dropouts = [keras.layers.Dropout(dropout_rate) for _ in hidden_layers]
        self.output_layer = keras.layers.Dense(embedding_dim, activation=None)
        self.l2_norm = keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1), name="user_l2_norm"
        )

    def call(self, user_idx: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.user_embedding(user_idx)
        for dense, drop in zip(self.dense_layers, self.dropouts):
            x = dense(x)
            x = drop(x, training=training)
        x = self.output_layer(x)
        return self.l2_norm(x)


class ItemTower(keras.layers.Layer):
    """Encodes movie_id + optional side features into a fixed-dim embedding."""

    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 64,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.3,
        l2: float = 0.001,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        hidden_layers = hidden_layers or [256, 128]

        self.item_embedding = keras.layers.Embedding(
            input_dim=n_items + 1,
            output_dim=embedding_dim,
            embeddings_regularizer=keras.regularizers.l2(l2),
            name="item_embedding",
        )
        self.dense_layers = [
            keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2),
            )
            for units in hidden_layers
        ]
        self.dropouts = [keras.layers.Dropout(dropout_rate) for _ in hidden_layers]
        self.output_layer = keras.layers.Dense(embedding_dim, activation=None)
        self.l2_norm = keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1), name="item_l2_norm"
        )

    def call(self, item_idx: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.item_embedding(item_idx)
        for dense, drop in zip(self.dense_layers, self.dropouts):
            x = dense(x)
            x = drop(x, training=training)
        x = self.output_layer(x)
        return self.l2_norm(x)


class TwoTowerModel(keras.Model):
    """Full two-tower model with retrieval score via dot product."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.3,
        l2: float = 0.001,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        hidden_layers = hidden_layers or [256, 128]

        self.user_tower = UserTower(
            n_users, embedding_dim, hidden_layers, dropout_rate, l2
        )
        self.item_tower = ItemTower(
            n_items, embedding_dim, hidden_layers, dropout_rate, l2
        )
        # Scale dot product to rating range [1, 5]
        self.score_scale = keras.layers.Dense(1, activation="sigmoid")

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        user_idx, item_idx = inputs
        user_emb = self.user_tower(user_idx, training=training)
        item_emb = self.item_tower(item_idx, training=training)
        # Dot product similarity, then scale to [0,1]
        dot = tf.reduce_sum(user_emb * item_emb, axis=1, keepdims=True)
        return self.score_scale(dot)

    def get_user_embedding(self, user_idx: tf.Tensor) -> tf.Tensor:
        return self.user_tower(user_idx, training=False)

    def get_item_embedding(self, item_idx: tf.Tensor) -> tf.Tensor:
        return self.item_tower(item_idx, training=False)

    def get_config(self) -> dict:
        return {
            "n_users": self.user_tower.user_embedding.input_dim - 1,
            "n_items": self.item_tower.item_embedding.input_dim - 1,
        }
