from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.feature_cross import CINLayer
from trainer.models.common.basic_layers import MLPLayer

from trainer.util.tools import ObjectDict


class xDeepFM(tfrs.Model):
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
        self.linear = tf.keras.experimental.LinearModel(
            kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001)
        )
        cin_layer_sizes = list(
            map(int, self.hparams.cin_layer_sizes.strip().split(","))
        )
        self.cin = tf.keras.Sequential(
            [
                CINLayer(
                    layer_sizes=cin_layer_sizes,
                    activation=self.hparams.cin_activation,
                    split_half=self.hparams.split_half,
                ),
                tf.keras.layers.Dense(1),
            ]
        )
        dnn_layer_sizes = list(
            map(int, self.hparams.dnn_layer_sizes.strip().split(","))
        )
        self.deep = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                MLPLayer(dnn_layer_sizes),
            ]
        )
        self.prediction = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001),
        )

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        deep_emb = self.deep_emb(features)
        return self.prediction(
            self.linear(self.wide_emb(features), training=training)
            + self.cin(deep_emb, training=training)
            + self.deep(deep_emb, training=training),
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
