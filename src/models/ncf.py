"""
Neural Collaborative Filtering (NCF) Model.

Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
as described in He et al., 2017: "Neural Collaborative Filtering".
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


class NeuralCollaborativeFiltering(keras.Model):
    """NCF model: GMF path + MLP path → NeuMF fusion layer."""

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
        hidden_layers = hidden_layers or [256, 128, 64]

        # GMF embeddings
        self.gmf_user_emb = keras.layers.Embedding(
            n_users + 1, embedding_dim,
            embeddings_regularizer=keras.regularizers.l2(l2),
            name="gmf_user",
        )
        self.gmf_item_emb = keras.layers.Embedding(
            n_items + 1, embedding_dim,
            embeddings_regularizer=keras.regularizers.l2(l2),
            name="gmf_item",
        )

        # MLP embeddings (separate embeddings improve performance)
        self.mlp_user_emb = keras.layers.Embedding(
            n_users + 1, embedding_dim,
            embeddings_regularizer=keras.regularizers.l2(l2),
            name="mlp_user",
        )
        self.mlp_item_emb = keras.layers.Embedding(
            n_items + 1, embedding_dim,
            embeddings_regularizer=keras.regularizers.l2(l2),
            name="mlp_item",
        )

        # MLP layers
        self.mlp_layers = []
        self.mlp_dropouts = []
        for units in hidden_layers:
            self.mlp_layers.append(
                keras.layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(l2),
                )
            )
            self.mlp_dropouts.append(keras.layers.Dropout(dropout_rate))

        # NeuMF output: concatenate GMF + MLP last hidden, then predict
        self.output_layer = keras.layers.Dense(1, activation="sigmoid", name="prediction")
        self.batch_norm = keras.layers.BatchNormalization()

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        user_idx, item_idx = inputs

        # GMF path: element-wise product
        gmf_u = self.gmf_user_emb(user_idx)
        gmf_i = self.gmf_item_emb(item_idx)
        gmf_out = gmf_u * gmf_i  # (batch, embedding_dim)

        # MLP path: concatenate → hidden layers
        mlp_u = self.mlp_user_emb(user_idx)
        mlp_i = self.mlp_item_emb(item_idx)
        mlp_x = tf.concat([mlp_u, mlp_i], axis=1)
        for dense, drop in zip(self.mlp_layers, self.mlp_dropouts):
            mlp_x = dense(mlp_x)
            mlp_x = self.batch_norm(mlp_x, training=training)
            mlp_x = drop(mlp_x, training=training)

        # Fusion
        fusion = tf.concat([gmf_out, mlp_x], axis=1)
        return self.output_layer(fusion)

    def get_config(self) -> dict:
        return {
            "n_users": self.gmf_user_emb.input_dim - 1,
            "n_items": self.gmf_item_emb.input_dim - 1,
        }
