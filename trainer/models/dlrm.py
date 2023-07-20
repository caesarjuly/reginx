from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.feature_cross import CINLayer
from trainer.models.common.basic_layers import DNNLayer

from trainer.util.tools import ObjectDict


class DLRM(tfrs.Model):
    def __init__(
        self, hparams: ObjectDict, deep_emb: tf.keras.Model, wide_emb: tf.keras.Model
    ):
        super().__init__()
        self.deep_emb = deep_emb
        self.wide_emb = wide_emb
        self.hparams = hparams
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.AUC()],
        )
        bottom_layer_sizes = list(
            map(int, self.hparams.bottom_layer_sizes.strip().split(","))
        )
        self.bottom_deep = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                DNNLayer(layer_sizes=bottom_layer_sizes),
            ]
        )
        self.feature_cross = tfrs.layers.feature_interaction.DotInteraction()
        top_layer_sizes = list(
            map(int, self.hparams.top_layer_sizes.strip().split(","))
        )
        self.top_deep = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                DNNLayer(layer_sizes=top_layer_sizes),
            ]
        )
        self.prediction = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(l2=0.001),
        )

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        dense_bottom_emb = self.bottom_deep(self.wide_emb(features), training=training)
        sparse_embs = tf.unstack(self.deep_emb(features), axis=1)
        cross_emb = self.feature_cross([dense_bottom_emb] + sparse_embs)
        return self.prediction(
            self.top_deep(
                tf.concat([dense_bottom_emb, cross_emb], axis=-1), training=training
            )
        )

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
