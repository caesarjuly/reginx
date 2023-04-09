import json
import os
from typing import Dict, List, Tuple

import tensorflow as tf
import tensorflow_recommenders as tfrs

from trainer.common.gcp import BUCKET_NAME, download_from_directory
from trainer.tasks.base import BaseTask
from trainer.models.factory import model_factory


class TwoTowerTrain(BaseTask):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)
        self.train_data, self.test_data, self.candidate_data = self.load_data()

    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        download_from_directory(BUCKET_NAME, self.hparams.train_data, "train")
        download_from_directory(BUCKET_NAME, self.hparams.test_data, "test")
        download_from_directory(BUCKET_NAME, self.hparams.candidate_data, "item")
        return (
            tf.data.Dataset.load("train"),
            tf.data.Dataset.load("test"),
            tf.data.Dataset.load("item"),
        )

    def run(self) -> Dict:
        query_model = model_factory.get_class(self.hparams.query_model)(
            self.hparams, self.meta
        )
        candidate_model = model_factory.get_class(self.hparams.candidate_model)(
            self.hparams, self.meta
        )
        two_tower_model = model_factory.get_class(self.hparams.base_model)
        self.model = two_tower_model(
            self.hparams, query_model, candidate_model, self.candidate_data
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.hparams.learning_rate)
        )
        train = self.train_data.batch(self.hparams.batch_size).shuffle(100_000).cache()

        test = self.test_data.batch(self.hparams.batch_size).shuffle(100_000).cache()
        uniform_negatives = (
            self.candidate_data.cache()
            .repeat()
            .shuffle(1_000)
            .batch(self.hparams.mixed_negative_batch_size)
        )
        train_with_mns = tf.data.Dataset.zip((train, uniform_negatives))

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.hparams.log_dir, histogram_freq=1
        )

        # Train.
        self.model.fit(
            train_with_mns,
            epochs=self.hparams.epochs,
            callbacks=[tensorboard_callback],
        )
        # evaluate
        return self.model.evaluate(test, return_dict=True)

    def save(self) -> None:
        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(self.model.query_model)
        # recommends movies out of the entire movies dataset.
        index.index_from_dataset(
            self.candidate_data.batch(self.hparams.batch_size).map(
                lambda x: (x[self.hparams.item_id_key], self.model.candidate_model(x))
            )
        )
        # https://github.com/tensorflow/tensorflow/issues/37439#issuecomment-596916472
        index.predict(self.test_data.take(10).batch(5))
        # Save the index.
        # https://github.com/tensorflow/models/issues/8990#issuecomment-1069733488
        tf.saved_model.save(index, self.hparams.model_dir)
