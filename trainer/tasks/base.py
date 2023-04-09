from abc import ABC, abstractmethod
import json
from typing import Tuple

import tensorflow as tf

from trainer.common.gcp import BUCKET_NAME, download_from_directory


class BaseTask(ABC):
    def __init__(self, hparams) -> None:
        self.hparams = hparams
        self.meta = self.load_meta()
        super().__init__()

    def load_meta(self):
        with open(self.hparams.meta_data, "r") as f:
            return json.load(f)

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def save(self):
        pass
