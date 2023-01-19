# rcnn_P2inP3out_P2zeroyouxiajiao256_finenet.py
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.layers.batch_norm import FrozenBatchNorm2d, get_norm

# from ..backbone import Backbone, build_backbone
# from ..postprocessing import detector_postprocess
# from ..proposal_generator import build_proposal_generator
# from ..roi_heads import build_roi_heads
# from .build import META_ARCH_REGISTRY


# # --add compressai framework
import sys
# sys.path.append("/media/data/liutie/VCM/rcnn/VCMbelle_0622")
# sys.path.append("/media/data/liutie/VCM/rcnn/VCMbelle_0622/VCM")
sys.path.append("/media/data/ccr/liutieCompressAI/")
sys.path.append("/media/data/ccr/liutieCompressAI/VCM/")
from compressai.datasets.parse_p2_feature import _joint_split_features
from examples.train_in_this import * #只用到了IRN_inference和main_cai
#
# main_cai()
# # --add compressai framework done

import functools
from torch.autograd import Variable
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from compressai.models import *
import random
import time
from torch.optim import Adam

# __all__ = ["GeneralizedRCNN", "ProposalNetwork"]
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_savepth = '/media/data/ccr/liutie_save/output/EXP_cheng2020anchor_256chinput_P2inP3outMSE_P2zeroyouxiajiao256_lambda1_N192_7imgtrainft9999_small5Wtrain_eachdnorm_finenet_09062230/model_0037999.pth'
path_savenewpth = '/media/data/ccr/liutie_save/output/EXP_cheng2020anchor_256chinput_P2inP3outMSE_P2zeroyouxiajiao256_lambda1_N192_7imgtrainft9999_small5Wtrain_eachdnorm_finenet_09062230/model_0037999_norcnn.pth'
net_belle = Cheng2020Anchor(N=192)
net_belle = net_belle.to(device)
net_belle.load_state_dict(torch.load(path_savepth), strict=False)
torch.save(net_belle.state_dict(), path_savenewpth)



