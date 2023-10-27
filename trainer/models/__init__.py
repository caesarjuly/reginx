import tensorflow_recommenders as tfrs

from trainer.util.tools import Factory
from trainer.models.basic_ranker import *
from trainer.models.deepfm import *
from trainer.models.fm import *
from trainer.models.mask_net import *
from trainer.models.two_tower import *
from trainer.models.wide_and_deep import *
from trainer.models.features.movielens import *
from trainer.models.features.criteo import *
from trainer.models.dcn import *
from trainer.models.xdeepfm import *
from trainer.models.autoint import *
from trainer.models.dlrm import *
from trainer.models.dcn_v2 import *
from trainer.models.mask_cn import *
from trainer.models.dual_mlp import *
from trainer.models.final_mlp import *
from trainer.models.transformer import *
from trainer.models.esmm import *
from trainer.models.shared_bottom import *
from trainer.models.mmoe import *
from trainer.models.shared_bottom_gn import *
from trainer.models.shared_bottom_uw import *
from trainer.models.shared_bottom_dwa import *
from trainer.models.shared_bottom_dtp import *
from trainer.models.shared_bottom_peltr import *

model_factory = Factory()
model_factory.register_all_subclasses(tfrs.Model)
model_factory.register(TransformerModel)
