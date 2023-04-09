import tensorflow as tf

from trainer.models.domain.movielens import MovieLensCandidateModel, MovieLensQueryModel, MovieLensModel
from trainer.models.two_tower import TwoTower
from trainer.models.basic_ranker import BasicRanker
from trainer.util.tools import Factory

model_factory = Factory()
model_factory.register(MovieLensCandidateModel.__name__, MovieLensCandidateModel)
model_factory.register(MovieLensQueryModel.__name__, MovieLensQueryModel)
model_factory.register(MovieLensModel.__name__, MovieLensModel)
model_factory.register(TwoTower.__name__, TwoTower)
model_factory.register(BasicRanker.__name__, BasicRanker)
