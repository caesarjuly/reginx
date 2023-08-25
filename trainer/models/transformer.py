from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.transformer import masked_accuracy, masked_loss, Transformer

from trainer.util.tools import ObjectDict


class TransformerModel(tf.keras.Model):
    def __init__(
        self,
        hparams: ObjectDict,
    ):
        super().__init__()
        self.hparams = hparams
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

    def call(self, inputs, training=False) -> tf.Tensor:
        src, target = inputs
        logits = self.transformer(src, target, training=training)
        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass
        return logits
