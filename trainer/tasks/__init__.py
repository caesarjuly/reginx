from trainer.util.tools import Factory
from trainer.tasks.candidate_retriever_train import *
from trainer.tasks.ranker_train import *
from trainer.tasks.wide_and_deep_ranker_train import *
from trainer.tasks.fm_ranker_train import *
from trainer.tasks.generate_fm_embedding import *

task_factory = Factory()
task_factory.register_all_subclasses(BaseTask)
