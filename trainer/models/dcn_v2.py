from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.basic_layers import MLPLayer

from trainer.util.tools import ObjectDict
from trainer.models.common.feature_cross import (
    CrossNetV2Layer,
    CrossNetSimpleMixLayer,
    CrossNetGatingMixLayer,
)


class DeepCrossNetworkV2(tfrs.Model):
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
        if self.hparams.cross_net == "simple_mix":
            self.cross_net = CrossNetSimpleMixLayer(layer_num=self.hparams.layer_num)
        elif self.hparams.cross_net == "gating_mix":
            self.cross_net = CrossNetGatingMixLayer(
                layer_num=self.hparams.layer_num,
                expert_num=self.hparams.expert_num,
                gate_func=self.hparams.gate_func,
                activation=self.hparams.activation,
            )
        else:
            self.cross_net = CrossNetV2Layer(layer_num=self.hparams.layer_num)
        layer_sizes = list(map(int, self.hparams.layer_sizes.strip().split(",")))
        self.dense = MLPLayer(layer_sizes)
        self.concat = tf.keras.layers.Concatenate()
        self.prediction = tf.keras.layers.Dense(1, "sigmoid")

    def call(self, features: Dict[Text, tf.Tensor], **kwargs) -> tf.Tensor:
        feat_emb = self.ranking_emb(features, **kwargs)
        if self.hparams.mode == "parallel":
            return self.prediction(
                self.concat(
                    [self.cross_net(feat_emb, **kwargs), self.dense(feat_emb, **kwargs)]
                )
            )
        else:
            return self.prediction(
                self.dense(self.cross_net(features, **kwargs), **kwargs)
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
