from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.basic_layers import DNNLayer
from trainer.models.common.feature_cross import FMLayer

from trainer.util.tools import ObjectDict


class DeepFM(tfrs.Model):
    def __init__(
        self,
        hparams: ObjectDict,
        deep_emb: tf.keras.Model,
        wide_emb: tf.keras.Model,
        sparse_linear_emb: tf.keras.Model = None,
    ):
        super().__init__()
        self.deep_emb = deep_emb
        self.wide_emb = wide_emb
        # optional, sparse features as 1d embedding to catch 1st order interaction
        self.sparse_linear_emb = sparse_linear_emb
        self.hparams = hparams
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.AUC()],
        )
        self.linear = tf.keras.experimental.LinearModel(
            kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001)
        )
        self.fm = FMLayer()
        layer_sizes = list(map(int, self.hparams.layer_sizes.strip().split(",")))
        self.deep = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                DNNLayer(layer_sizes),
            ]
        )
        self.prediction = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001),
        )

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        deep_emb = self.deep_emb(features, training=training)
        logits = (
            self.linear(self.wide_emb(features), training=training)
            + self.fm(deep_emb, training=training)
            + self.deep(deep_emb, training=training)
        )
        # add sparse 1d embedding to linear weights
        if self.sparse_linear_emb:
            logits += tf.reduce_sum(
                self.sparse_linear_emb(features, training=training),
                axis=-1,
                keepdims=True,
            )
        return self.prediction(logits)

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
