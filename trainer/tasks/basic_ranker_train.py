import json
import os
from typing import Dict, List, Tuple
import math

import tensorflow as tf

from trainer.common.gcp import BUCKET_NAME, download_from_directory
from trainer.tasks.base import BaseTask
from trainer.models.factory import model_factory


class BasicRankerTrain(BaseTask):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)
        self.train_data, self.test_data = self.load_data()

    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        download_from_directory(BUCKET_NAME, self.hparams.train_data, "train")
        download_from_directory(BUCKET_NAME, self.hparams.test_data, "test")
        return (
            tf.data.Dataset.load("train"),
            tf.data.Dataset.load("test"),
        )

    def run(self) -> Dict:
        ranking_model = model_factory.get_class(self.hparams.ranking_model)(
            self.hparams, self.meta
        )
        base_ranker = model_factory.get_class(self.hparams.base_ranker)
        self.model = base_ranker(self.hparams, ranking_model)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.hparams.learning_rate, decay_steps=1000, decay_rate=0.9
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        )
        train = self.train_data.batch(self.hparams.batch_size).shuffle(1_000).cache()

        test = self.test_data.batch(self.hparams.batch_size).shuffle(1_000).cache()
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.hparams.log_dir, histogram_freq=1
        )

        # Train.
        self.model.fit(
            train,
            epochs=self.hparams.epochs,
            callbacks=[tensorboard_callback],
        )
        # evaluate
        return self.model.evaluate(test, return_dict=True)

    def save(self) -> None:
        # https://github.com/tensorflow/tensorflow/issues/37439#issuecomment-596916472
        data = self.test_data.take(20).batch(20)
        for i in data.as_numpy_iterator():
            print(i["user_rating"])
        result = self.model.predict(data)
        print([i[0] for i in result])
        # Save the index.
        # https://github.com/tensorflow/models/issues/8990#issuecomment-1069733488
        tf.saved_model.save(self.model, self.hparams.model_dir)
