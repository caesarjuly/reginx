from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.basic_layers import MLPLayer
from trainer.models.common.feature_cross import (
    FeatureSelectionLayer,
    InteractionAggregationLayer,
)

from trainer.util.tools import ObjectDict


class FinalMLP(tfrs.Model):
    """FinalMLP consists of two feature selection layer and a multi-head bilinear fusion layer"""

    def __init__(
        self,
        hparams: ObjectDict,
        sparse_emb: tf.keras.Model,
        dense_emb: tf.keras.Model,
    ):
        super().__init__()
        self.sparse_emb = sparse_emb
        self.dense_emb = dense_emb
        self.hparams = hparams
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.AUC()],
        )

        hidden_dims = list(map(int, self.hparams.hidden_dims.strip().split(",")))
        layer_sizes1 = list(map(int, self.hparams.layer_sizes1.strip().split(",")))
        self.dense1 = MLPLayer(layer_sizes1)
        self.fs_context1 = list(self.hparams.fs_context1.strip().split(","))
        self.fs1 = FeatureSelectionLayer(hidden_dims=hidden_dims)
        layer_sizes2 = list(map(int, self.hparams.layer_sizes2.strip().split(",")))
        self.dense2 = MLPLayer(layer_sizes2)
        self.fs_context2 = list(self.hparams.fs_context2.strip().split(","))
        self.fs2 = FeatureSelectionLayer(hidden_dims=hidden_dims)

        self.interaction = InteractionAggregationLayer(head_num=self.hparams.head_num)
        self.prediction = tf.keras.layers.Dense(1, "sigmoid")

    def call(self, features: Dict[Text, tf.Tensor], **kwargs) -> tf.Tensor:
        # feature embedding
        dense_emb = self.dense_emb(features, **kwargs)
        sparse_emb = self.sparse_emb(features, **kwargs)
        concat_sparse_emb = tf.concat(tf.unstack(sparse_emb, axis=1), axis=-1)
        feat_emb = tf.concat([dense_emb, concat_sparse_emb], axis=-1)

        # feature selection 1
        feat_ctx_emb1 = self.dense_emb(features, self.fs_context1, **kwargs)
        feat_selection1 = self.fs1((feat_emb, feat_ctx_emb1))

        # feature selection 2
        feat_ctx_emb2 = tf.concat(
            tf.unstack(self.sparse_emb(features, self.fs_context2, **kwargs), axis=1),
            axis=-1,
        )
        feat_selection2 = self.fs2((feat_emb, feat_ctx_emb2))
        interaction = self.interaction(
            (
                self.dense1(feat_selection1, **kwargs),
                self.dense2(feat_selection2, **kwargs),
            )
        )
        return self.prediction(interaction)

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        labels = features[self.hparams.label]
        rating_predictions = self(features, training=training)

        # The task computes the loss and the metrics.
        return self.task(
            labels=labels,
            predictions=rating_predictions,
            training=training,
        )
