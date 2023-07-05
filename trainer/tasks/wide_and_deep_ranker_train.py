import json
import os
from typing import Dict, List, Tuple
import math

import tensorflow as tf

from trainer.common.gcp import BUCKET_NAME, download_from_directory
from trainer.tasks.ranker_train import RankerTrain
from trainer.models import model_factory


class WideAndDeepRankerTrain(RankerTrain):
    def run(self) -> Dict:
        ranking_embs = [
            model_factory.get_class(emb.strip())(self.meta)
            for emb in self.hparams.ranking_emb.split(",")
        ]
        ranker = model_factory.get_class(self.hparams.ranker)
        self.model = ranker(self.hparams, *ranking_embs)
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
        self.model.summary()
        # evaluate
        return self.model.evaluate(test, return_dict=True)
