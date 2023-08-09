from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.basic_layers import DNNLayer
from trainer.models.common.feature_cross import MaskBlock

from trainer.util.tools import ObjectDict


class MaskNet(tfrs.Model):
    """MaskNet have two modes, parralel and serial"""

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
            layer_sizes = list(map(int, self.hparams.layer_sizes.strip().split(",")))
            self.dense = tf.keras.Sequential(
                [
                    DNNLayer(layer_sizes),
                    tf.keras.layers.Dense(
                        1,
                        activation="sigmoid",
                        kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001),
                    ),
                ]
            )
        else:
            self.dense = tf.keras.layers.Dense(
                1,
                "sigmoid",
                kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001),
            )

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        feat_emb = self.ranking_emb(features, training)
        # In parallel mode, both inputs are feature embedding
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
            # In serial mode, the feature embedding is for mask calculation, the hidden embedding is the input for next MaskBlock
            hidden_emb = feat_emb
            for mask_block in self.mask_blocks:
                hidden_emb = mask_block((feat_emb, hidden_emb))
            return self.dense(hidden_emb)

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
