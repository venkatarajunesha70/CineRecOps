"""
Hybrid Recommender — combines collaborative filtering with content-based features.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


class HybridRecommender(keras.Model):
    """Hybrid model that fuses ID-based embeddings with dense side features."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_user_features: int,
        n_item_features: int,
        embedding_dim: int = 64,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.3,
        l2: float = 0.001,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        hidden_layers = hidden_layers or [256, 128, 64]

        # ID embeddings
        self.user_emb = keras.layers.Embedding(
            n_users + 1, embedding_dim,
            embeddings_regularizer=keras.regularizers.l2(l2),
        )
        self.item_emb = keras.layers.Embedding(
            n_items + 1, embedding_dim,
            embeddings_regularizer=keras.regularizers.l2(l2),
        )

        # Side-feature encoders
        self.user_feature_enc = keras.Sequential([
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(embedding_dim, activation="relu"),
        ], name="user_feature_encoder")

        self.item_feature_enc = keras.Sequential([
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(embedding_dim, activation="relu"),
        ], name="item_feature_encoder")

        # Fusion MLP
        self.fusion_layers = []
        self.fusion_dropouts = []
        for units in hidden_layers:
            self.fusion_layers.append(
                keras.layers.Dense(units, activation="relu",
                                   kernel_regularizer=keras.regularizers.l2(l2))
            )
            self.fusion_dropouts.append(keras.layers.Dropout(dropout_rate))

        self.output_layer = keras.layers.Dense(1, activation="sigmoid")

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        user_idx, item_idx, user_feats, item_feats = inputs

        # ID embeddings
        u_emb = self.user_emb(user_idx)
        i_emb = self.item_emb(item_idx)

        # Side-feature encodings
        u_feat = self.user_feature_enc(user_feats, training=training)
        i_feat = self.item_feature_enc(item_feats, training=training)

        # Fuse all representations
        x = tf.concat([u_emb, i_emb, u_feat, i_feat], axis=1)
        for dense, drop in zip(self.fusion_layers, self.fusion_dropouts):
            x = dense(x)
            x = drop(x, training=training)

        return self.output_layer(x)

    def get_config(self) -> dict:
        return {
            "n_users": self.user_emb.input_dim - 1,
            "n_items": self.item_emb.input_dim - 1,
        }
