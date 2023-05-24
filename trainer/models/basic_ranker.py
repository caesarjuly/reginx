from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs

from trainer.util.tools import ObjectDict


class BasicRanker(tfrs.Model):
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
        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(1, "sigmoid"),
            ]
        )

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        return self.dense(self.ranking_emb(features, training), training)

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        labels = tf.expand_dims(
            tf.where(features[self.hparams.label] > 0, 1, 0), axis=-1
        )
        sample_weight = (
            tf.expand_dims(
                tf.where(
                    features[self.hparams.label] > 0, features[self.hparams.label], 1
                ),
                axis=-1,
            )
            if training
            else None
        )

        rating_predictions = self(features, training=training)

        # The task computes the loss and the metrics.
        return self.task(
            labels=labels,
            predictions=rating_predictions,
            sample_weight=sample_weight,
            training=training,
        )
