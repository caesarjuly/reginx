from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.feature_cross import MultiHeadSelfAttentionLayer
from trainer.models.common.basic_layers import MLPLayer

from trainer.util.tools import ObjectDict


class AutoInt(tfrs.Model):
    def __init__(self, hparams: ObjectDict, rank_emb: tf.keras.Model):
        super().__init__()
        self.rank_emb = rank_emb
        self.hparams = hparams
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.AUC()],
        )
        self.attentions = tf.keras.Sequential(
            [
                MultiHeadSelfAttentionLayer(
                    head_dim=hparams.head_dim, head_num=hparams.head_num
                )
                for _ in range(hparams.att_layer_num)
            ]
            + [tf.keras.layers.Flatten()]
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
        deep_emb = self.rank_emb(features)
        return self.prediction(
            tf.concat(
                [
                    self.attentions(deep_emb, training=training),
                    self.deep(deep_emb, training=training),
                ],
                axis=-1,
            ),
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
