from abc import ABC, abstractmethod
import json


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


class NLPTask(ABC):
    def __init__(self, hparams) -> None:
        self.hparams = hparams
        super().__init__()

    @abstractmethod
    def run(self):
        pass
