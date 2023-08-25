import json
import os
from typing import Dict, List, Tuple
import math

import tensorflow as tf

from trainer.common.gcp import BUCKET_NAME, download_from_directory
from trainer.models.common.transformer import (
    CustomSchedule,
    masked_accuracy,
    masked_loss,
)
from trainer.models import model_factory
from trainer.tasks.base import NLPTask


class TransformerTrain(NLPTask):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)
        self.train_data, self.test_data = self.load_data()

    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        download_from_directory(BUCKET_NAME, self.hparams.train_data, "/tmp/train")
        download_from_directory(BUCKET_NAME, self.hparams.test_data, "/tmp/test")
        return (
            tf.data.experimental.load(
                "/tmp/train",
                compression="GZIP",
                reader_func=lambda datasets: datasets.interleave(
                    lambda x: x, num_parallel_calls=tf.data.AUTOTUNE
                ),
            ),
            tf.data.experimental.load(
                "/tmp/test",
                compression="GZIP",
                reader_func=lambda datasets: datasets.interleave(
                    lambda x: x, num_parallel_calls=tf.data.AUTOTUNE
                ),
            ),
        )

    def run(self) -> Dict:
        model = model_factory.get_class(self.hparams.model)
        self.model = model(self.hparams)
        learning_rate = CustomSchedule(self.hparams.model_dim)
        self.model.compile(
            loss=masked_loss,
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-9,
            ),
            metrics=[masked_accuracy],
        )
        train = (
            self.train_data.batch(self.hparams.batch_size)
            .shuffle(1_000)
            .prefetch(tf.data.AUTOTUNE)
        )

        test = (
            self.test_data.batch(self.hparams.batch_size)
            .shuffle(1_000)
            .prefetch(tf.data.AUTOTUNE)
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=f"/tmp/{self.hparams.log_dir}",
            histogram_freq=1,
            # profile_batch=(10, 20),
        )

        # Train.
        self.model.fit(
            train,
            epochs=self.hparams.epochs,
            validation_data=test,
            callbacks=[tensorboard_callback],
        )
        self.model.summary()
        # evaluate
        return self.model.evaluate(test, return_dict=True)

    def save(self) -> None:
        # https://github.com/tensorflow/tensorflow/issues/37439#issuecomment-596916472
        data = self.test_data.take(20).batch(20)
        for i in data.as_numpy_iterator():
            print(i[1])
        result = self.model.predict(data)
        print([i[0] for i in result])
        # Save the index.
        # https://github.com/tensorflow/models/issues/8990#issuecomment-1069733488
        tf.saved_model.save(self.model, f"/tmp/{self.hparams.model_dir}")
