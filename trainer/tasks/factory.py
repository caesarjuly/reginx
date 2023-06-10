from trainer.tasks.base import BaseTask

from trainer.tasks.candidate_retriever_train import CandidateRetrieverTrain
from trainer.tasks.ranker_train import RankerTrain
from trainer.tasks.wide_and_deep_ranker_train import WideAndDeepRankerTrain
from trainer.util.tools import Factory

task_factory = Factory()
task_factory.register_all_subclasses(BaseTask)
