import tensorflow as tf

from trainer.models.domain.movielens import (
    MovieLensCandidateEmb,
    MovieLensQueryEmb,
    MovieLensRankingEmb,
    MovieLensWideEmb,
)
from trainer.models.two_tower import TwoTower
from trainer.models.basic_ranker import BasicRanker
from trainer.models.mask_net import MaskNet
from trainer.models.wide_and_deep import WideAndDeep
from trainer.util.tools import Factory

model_factory = Factory()
model_factory.register(MovieLensCandidateEmb.__name__, MovieLensCandidateEmb)
model_factory.register(MovieLensQueryEmb.__name__, MovieLensQueryEmb)
model_factory.register(MovieLensRankingEmb.__name__, MovieLensRankingEmb)
model_factory.register(MovieLensWideEmb.__name__, MovieLensWideEmb)
model_factory.register(TwoTower.__name__, TwoTower)
model_factory.register(BasicRanker.__name__, BasicRanker)
model_factory.register(MaskNet.__name__, MaskNet)
model_factory.register(WideAndDeep.__name__, WideAndDeep)
