from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.transformer import masked_accuracy, masked_loss, Transformer

from trainer.util.tools import ObjectDict


class TransformerModel(tfrs.Model):
    def __init__(
        self,
        hparams: ObjectDict,
        ranking_emb: tf.keras.Model,
    ):
        super().__init__()
        self.ranking_emb = ranking_emb
        self.hparams = hparams
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=masked_loss,
            metrics=[masked_accuracy],
        )
        self.transformer = Transformer(
            self.hparams.src_vocab_size,
            self.hparams.target_vocab_size,
            self.hparams.seq_length,
            layer_num=self.hparams.layer_num,
            model_dim=self.hparams.model_dim,
            ff_dim=self.hparams.ff_dim,
            dropout=self.hparams.dropout,
            head_num=self.hparams.head_num,
        )

    def call(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        return self.transformer(features["src"], features["target"], training=training)

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
