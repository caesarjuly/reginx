from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.basic_layers import DNNLayer
from trainer.models.common.feature_cross import CrossNetV2Layer

from trainer.util.tools import ObjectDict


class CrossNetBlock(tf.keras.layers.Layer):
    """MaskBlock combine CrossNetV2 with LayerNorm and FC layer"""

    def __init__(
        self,
        cross_layer_num: int = 1,
        mask_block_dim: int = 64,
    ):
        super().__init__()
        self.ln = tf.keras.layers.LayerNormalization()
        self.relu = tf.keras.layers.Activation("relu")
        self.dense = tf.keras.layers.Dense(mask_block_dim)
        self.cn = CrossNetV2Layer(layer_num=cross_layer_num)

    def call(self, inputs, training=False):
        crossed_emb = self.cn(inputs, training=training)
        return self.relu(self.ln(self.dense(crossed_emb)))


class MaskCN(tfrs.Model):
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
        cross_layer_nums = list(
            map(int, self.hparams.cross_layer_nums.strip().split(","))
        )
        self.mask_cns = [
            CrossNetBlock(cross_layer_num=i, mask_block_dim=self.hparams.mask_block_dim)
            for i in cross_layer_nums
        ]
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

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        feat_emb = self.ranking_emb(features, training)
        block_out = []
        for cross_net in self.mask_cns:
            block_out.append(cross_net(feat_emb))
        return self.dense(tf.concat(block_out, -1))

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
