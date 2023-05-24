from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs

from trainer.util.tools import ObjectDict


class InstanceGuidedMask(tf.keras.layers.Layer):
    def __init__(
        self,
        output_dim: int,
        reduction_ratio: float = 2.0,
    ):
        super().__init__()
        self.aggregation = tf.keras.layers.Dense(output_dim * reduction_ratio)
        self.relu = tf.keras.layers.Activation("relu")
        self.projection = tf.keras.layers.Dense(output_dim)

    def call(self, feat_emb, training=False):
        return self.projection(self.relu(self.aggregation(feat_emb)))


class MaskBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        hparams: ObjectDict,
    ):
        super().__init__()
        self.hparams = hparams
        self.ln = tf.keras.layers.LayerNormalization()
        self.relu = tf.keras.layers.Activation("relu")

    def build(self, input_shape: tf.Tensor):
        _, hidden_emb_shape = input_shape
        self.instance_guided_mask = InstanceGuidedMask(
            output_dim=hidden_emb_shape[-1],
            reduction_ratio=self.hparams.reduction_ratio,
        )
        self.dense = tf.keras.layers.Dense(self.hparams.mask_block_dim)

    def call(self, inputs, training=False):
        feat_emb, hidden_emb = inputs
        masked_emb = self.instance_guided_mask(feat_emb, training=training) * hidden_emb
        return self.relu(self.ln(self.dense(masked_emb)))


class MaskNet(tfrs.Model):
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
        self.mask_blocks = [
            MaskBlock(hparams=hparams) for _ in range(self.hparams.mask_block_num)
        ]
        if self.hparams.mode == "parallel":
            self.dense = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(1, "sigmoid"),
                ]
            )
        else:
            self.dense = tf.keras.layers.Dense(1, "sigmoid")

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        feat_emb = self.ranking_emb(features, training)
        if self.hparams.mode == "parallel":
            block_out = []
            for mask_block in self.mask_blocks:
                block_out.append(
                    mask_block(
                        (
                            feat_emb,
                            feat_emb,
                        )
                    )
                )
                return self.dense(tf.concat(block_out, -1))
        else:
            hidden_emb = feat_emb
            for mask_block in self.mask_blocks:
                hidden_emb = mask_block((feat_emb, hidden_emb))
            return self.dense(hidden_emb)

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        labels = tf.expand_dims(
            tf.where(features[self.hparams.label] > 0, 1, 0), axis=-1
        )

        rating_predictions = self(features, training)

        # The task computes the loss and the metrics.
        return self.task(
            labels=labels,
            predictions=rating_predictions,
            training=training,
        )
