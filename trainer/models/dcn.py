from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.basic_layers import DNNLayer

from trainer.util.tools import ObjectDict
from trainer.models.common.feature_cross import CrossNetLayer


class DeepCrossNetwork(tfrs.Model):
    """DeepCrossNetwork consists of a cross net work and a deep dense net work"""

    def __init__(
        self,
        hparams: ObjectDict,
        ranking_emb: tf.keras.Model,
    ):
        super().__init__()
        self.ranking_emb = ranking_emb
        self.hparams = hparams
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.AUC()],
        )
        self.cross_net = CrossNetLayer(layer_num=self.hparams.layer_num)
        layer_sizes = list(map(int, self.hparams.layer_sizes.strip().split(",")))
        self.dense = DNNLayer(layer_sizes)
        self.concat = tf.keras.layers.Concatenate()
        self.prediction = tf.keras.layers.Dense(1, "sigmoid")

    def call(self, features: Dict[Text, tf.Tensor], **kwargs) -> tf.Tensor:
        feat_emb = self.ranking_emb(features, **kwargs)
        return self.prediction(
            self.concat(
                [self.cross_net(feat_emb, **kwargs), self.dense(feat_emb, **kwargs)]
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
