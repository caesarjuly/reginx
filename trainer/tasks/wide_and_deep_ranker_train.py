import json
import os
from typing import Dict, List, Tuple
import math

import tensorflow as tf

from trainer.common.gcp import BUCKET_NAME, download_from_directory
from trainer.tasks.ranker_train import RankerTrain
from trainer.models.factory import model_factory


class WideAndDeepRankerTrain(RankerTrain):
    def run(self) -> Dict:
        deep_emb = model_factory.get_class(self.hparams.deep_emb)(self.meta)
        wide_emb = model_factory.get_class(self.hparams.wide_emb)(self.meta)
        ranker = model_factory.get_class(self.hparams.ranker)
        self.model = ranker(self.hparams, deep_emb, wide_emb)
        self.model.compile(optimizer=["ftrl", "adam"])

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
