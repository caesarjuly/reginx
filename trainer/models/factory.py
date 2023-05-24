import tensorflow as tf

from trainer.models.domain.movielens import (
    MovieLensCandidateEmb,
    MovieLensQueryEmb,
    MovieLensRankingEmb,
)
from trainer.models.two_tower import TwoTower
from trainer.models.basic_ranker import BasicRanker
from trainer.models.mask_net import MaskNet
from trainer.util.tools import Factory

model_factory = Factory()
model_factory.register(MovieLensCandidateEmb.__name__, MovieLensCandidateEmb)
model_factory.register(MovieLensQueryEmb.__name__, MovieLensQueryEmb)
model_factory.register(MovieLensRankingEmb.__name__, MovieLensRankingEmb)
model_factory.register(TwoTower.__name__, TwoTower)
model_factory.register(BasicRanker.__name__, BasicRanker)
model_factory.register(MaskNet.__name__, MaskNet)
