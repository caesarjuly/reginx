from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.fm import FMLayer

from trainer.util.tools import ObjectDict


# a wrapper for easy export
class LinearModelWrapper(tfrs.Model):
    def __init__(self, emb_model: tfrs.Model):
        super().__init__()
        self.emb_model = emb_model
        self.linear_model = tf.keras.experimental.LinearModel(
            kernel_regularizer=tf.keras.regularizers.l2(l2=0.001), use_bias=False
        )

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        return self.linear_model(
            self.emb_model(features, training=training), training=training
        )


class FM(tfrs.Model):
    def __init__(
        self,
        hparams: ObjectDict,
        wide_user_emb: tfrs.Model,
        wide_item_emb: tfrs.Model,
        deep_user_emb: tfrs.Model,
        deep_item_emb: tfrs.Model,
    ):
        super().__init__()
        self.deep_user_emb = deep_user_emb
        self.deep_item_emb = deep_item_emb
        self.hparams = hparams
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.AUC()],
        )
        self.linear_user = LinearModelWrapper(wide_user_emb)
        self.linear_item = LinearModelWrapper(wide_item_emb)
        self.bias = tf.Variable(0.0, trainable=True)
        self.fm = FMLayer()
        self.activation = tf.keras.layers.Activation("sigmoid")

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # concat the user and item features and feed to fm layer
        deep_emb = tf.concat(
            [self.deep_user_emb(features), self.deep_item_emb(features)], axis=1
        )
        return self.activation(
            self.linear_user(features, training=training)
            + self.linear_item(features, training=training)
            + self.bias
            + self.fm(deep_emb, training=training)
        )

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        labels = tf.expand_dims(
            tf.where(features[self.hparams.label] > 3, 1, 0), axis=-1
        )
        rating_predictions = self(features, training=training)

        # The task computes the loss and the metrics.
        return self.task(
            labels=labels,
            predictions=rating_predictions,
            training=training,
        )

    def get_models(self):
        return self.linear_user, self.linear_item

    def get_fm_emb(self):
        return (
            self.deep_user_emb,
            self.deep_item_emb,
        )
