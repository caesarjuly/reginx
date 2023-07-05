from typing import Dict
import tensorflow as tf
import tensorflow_recommenders as tfrs
from trainer.models.common.basic_layers import DNNLayer

from trainer.models.common.sampling_bias_correction import SamplingBiasCorrection
from trainer.util.tools import ObjectDict


class DenseModel(tf.keras.Model):
    def __init__(
        self,
        emb_model: tf.keras.Model,
    ):
        super().__init__()
        self.deep = (DNNLayer(layer_sizes=(128, 64)),)
        self.emb_model = emb_model

    def call(self, inputs, training=False):
        embeddings = self.dense(self.emb_model(inputs, training))
        return tf.math.l2_normalize(embeddings, -1)


class TwoTower(tfrs.Model):
    def __init__(
        self,
        hparams: ObjectDict,
        query_emb: tf.keras.Model,
        candidate_emb: tf.keras.Model,
        items: tf.data.Dataset,
    ):
        super().__init__()
        self.query_model = DenseModel(query_emb)
        self.candidate_model = DenseModel(candidate_emb)
        self.hparams = hparams
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=items.batch(128).map(
                    lambda x: (x[self.hparams.item_id_key], self.candidate_model(x))
                ),
            ),
            remove_accidental_hits=True,
            temperature=self.hparams.temperature,
        )
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.sampling_bias = SamplingBiasCorrection()

    def compute_loss(self, inputs, training=False) -> tf.Tensor:
        if training:
            self.global_step.assign_add(1)
            features, extra_items = inputs
            user_embeddings = self.query_model(features, training)
            candidates_embeddings = self.candidate_model(features, training)
            negatives_embeddings = self.candidate_model(extra_items, training)
            # we cannot turn on the topK metrics calculation for training when there is extra negatives, need modification on the Retrieval call function
            # true_candidate_ids=candidate_ids[:tf.shape(query_embeddings)[0]])
            candidate_ids = tf.concat(
                [
                    features[self.hparams.item_id_key],
                    extra_items[self.hparams.item_id_key],
                ],
                axis=-1,
            )
            candidate_embeddings = tf.concat(
                [candidates_embeddings, negatives_embeddings], axis=0
            )
            candidate_sampling_probability = self.sampling_bias(
                self.global_step, candidate_ids
            )
        else:
            user_embeddings = self.query_model(inputs)
            candidate_embeddings = self.candidate_model(inputs)
            candidate_sampling_probability = None
            candidate_ids = inputs[self.hparams.item_id_key]
        return self.task(
            user_embeddings,
            candidate_embeddings,
            candidate_sampling_probability=candidate_sampling_probability,
            candidate_ids=candidate_ids,
            compute_metrics=not training,
        )
