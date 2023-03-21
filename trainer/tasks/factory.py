
from trainer.tasks.base import BaseTask

from trainer.tasks.two_tower_train import TwoTowerTrain
from trainer.util.tools import Factory

task_factory = Factory()
task_factory.register_all_subclasses(BaseTask)
