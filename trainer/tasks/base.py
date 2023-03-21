from abc import ABC, abstractmethod


class BaseTask(ABC):

    def __init__(self, hparams) -> None:
        self.hparams = hparams
        super().__init__()

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def save(self):
        pass
