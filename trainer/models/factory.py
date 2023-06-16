import tensorflow as tf

from trainer.models.domain.movielens import (
    MovieLensCandidateEmb,
    MovieLensQueryEmb,
    MovieLensRankingEmb,
    MovieLensWideEmb,
    MovieLensFMWideEmb,
    MovieLensFMDeepEmb,
)
from trainer.models.two_tower import TwoTower
from trainer.models.basic_ranker import BasicRanker
from trainer.models.mask_net import MaskNet
from trainer.models.wide_and_deep import WideAndDeep
from trainer.models.fm import FM
from trainer.models.deepfm import DeepFM
from trainer.util.tools import Factory

model_factory = Factory()
model_factory.register(MovieLensCandidateEmb.__name__, MovieLensCandidateEmb)
model_factory.register(MovieLensQueryEmb.__name__, MovieLensQueryEmb)
model_factory.register(MovieLensRankingEmb.__name__, MovieLensRankingEmb)
model_factory.register(MovieLensWideEmb.__name__, MovieLensWideEmb)
model_factory.register(MovieLensFMWideEmb.__name__, MovieLensFMWideEmb)
model_factory.register(MovieLensFMDeepEmb.__name__, MovieLensFMDeepEmb)
model_factory.register(TwoTower.__name__, TwoTower)
model_factory.register(BasicRanker.__name__, BasicRanker)
model_factory.register(MaskNet.__name__, MaskNet)
model_factory.register(WideAndDeep.__name__, WideAndDeep)
model_factory.register(FM.__name__, FM)
model_factory.register(DeepFM.__name__, DeepFM)
