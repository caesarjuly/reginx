from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.basic_layers import DNNLayer

from trainer.util.tools import ObjectDict


class ESMM(tfrs.Model):
    def __init__(
        self,
        hparams: ObjectDict,
        ranking_emb: tf.keras.Model,
    ):
        super().__init__()
        self.ranking_emb = ranking_emb
        self.hparams = hparams
        self.pctr_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.AUC()],
        )
        self.pctcvr_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.AUC()],
        )
        self.pctr_weight = hparams.pctr_weight
        self.pctcvr_weight = hparams.pctcvr_weight
        self.dense1 = tf.keras.Sequential(
            [
                DNNLayer(),
                tf.keras.layers.Dense(1, "sigmoid"),
            ]
        )
        self.dense2 = tf.keras.Sequential(
            [
                DNNLayer(),
                tf.keras.layers.Dense(1, "sigmoid"),
            ]
        )

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        shared_emb = self.ranking_emb(features, training)
        pctr = self.dense1(shared_emb, training)
        pcvr = self.dense2(shared_emb, training)
        return pctr, pcvr

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        ctr_label = tf.expand_dims(
            tf.where(features[self.hparams.label] > 0, 1, 0), axis=-1
        )
        ctcvr_label = tf.expand_dims(
            tf.where(features[self.hparams.label] > 3, 1, 0), axis=-1
        )

        pctr, pcvr = self(features, training=training)
        pctcvr = pctr * pcvr

        # pctr loss
        pctr_loss = self.pctr_task(
            labels=ctr_label,
            predictions=pctr,
            training=training,
        )
        # pctcvr loss
        pctcvr_loss = self.pctcvr_task(
            labels=ctcvr_label,
            predictions=pctcvr,
            training=training,
        )

        return self.pctr_weight * pctr_loss + self.pctcvr_weight * pctcvr_loss
