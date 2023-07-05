from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.basic_layers import DNNLayer

from trainer.util.tools import ObjectDict


class WideAndDeepTFRS(tfrs.Model):
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs, training=True)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

            linear_vars = self.wide.trainable_variables
            dnn_vars = self.deep.trainable_variables
            linear_grads, dnn_grads = tape.gradient(loss, (linear_vars, dnn_vars))

            linear_optimizer = self.optimizer[0]
            dnn_optimizer = self.optimizer[1]
            linear_optimizer.apply_gradients(zip(linear_grads, linear_vars))
            dnn_optimizer.apply_gradients(zip(dnn_grads, dnn_vars))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics


class WideAndDeep(WideAndDeepTFRS):
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
        self.wide = tf.keras.experimental.LinearModel(
            kernel_regularizer=tf.keras.regularizers.l2(l2=0.001)
        )
        self.deep = DNNLayer(output_logits=True)
        self.activation = tf.keras.layers.Activation("sigmoid")

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        return self.activation(
            self.deep(self.deep_emb(features), training=training)
            + self.wide(self.wide_emb(features), training=training)
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
