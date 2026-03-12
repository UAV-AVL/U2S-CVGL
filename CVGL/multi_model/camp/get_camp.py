import os
import time
import cv2
import math
import shutil
import sys
import torch
import argparse
import numpy as np

from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup

# from sample4geo.dataset.university import U1652DatasetEval, U1652DatasetTrain, get_transforms
# from sample4geo.utils import setup_system, Logger
# from sample4geo.trainer import train
# from sample4geo.evaluate.university import evaluate

import warnings
warnings.filterwarnings('ignore')
# from sample4geo.model import TimmModel
from torch.utils.tensorboard import SummaryWriter

from multi_model.camp.sample4geo.hand_convnext.model import make_model

start_time = time.time()

@dataclass
class Configuration:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train and Test on University-1652')

        # Added for your modification
        # parser.add_argument('--model', default='convnext_base.fb_in22k_ft_in1k_384', type=str, help='backbone model')
        parser.add_argument('--model', default='convnext_tiny', type=str, help='backbone model')
        # parser.add_argument('--model', default='convnext_femto_ols.d1_in1k', type=str, help='backbone model')
        # parser.add_argument('--model', default='convnext_nano.in12k_ft_in1k', type=str, help='backbone model')
        # parser.add_argument('--model', default='convnext_base_22k_1k_384', type=str, help='backbone model')
        # parser.add_argument('--model', default='convnext_pico.d1_in1k', type=str, help='backbone model')
        # parser.add_argument('--model', default='convnext_femto.d1_in1k', type=str, help='backbone model')
        parser.add_argument('--backbone_model', default=1, type=int, help='use modified backbone')

        # parser.add_argument('--img_size', default=256, type=int, help='input image size')
        parser.add_argument('--views', default=2, type=int, help='only supports 2 branches retrieval')
        parser.add_argument('--record', default=True, type=bool, help='use tensorboard to record training procedure')

        # -- ## 移动到前面方便修改权重
        parser.add_argument('--only_test', default=True, type=bool, help='use pretrained model to test')
        parser.add_argument('--only_draw_heat', default=False, type=bool, help='use pretrained model to test')
        # parser.add_argument('--ckpt_path',
        #                     default='multi_model/weights/tiny_256_3epoch.pth',
        #                     type=str, help='path to pretrained checkpoint file')

        # Model Config
        parser.add_argument('--nclasses', default=701, type=int, help='U-1652场景的类别数')
        parser.add_argument('--block', default=2, type=int)
        parser.add_argument('--triplet_loss', default=0.3, type=float)
        parser.add_argument('--resnet', default=False, type=bool)

        # Our tricks
        # parser.add_argument('--weight_infonce', default=1.0, type=float)
        # parser.add_argument('--weight_triplet', default=0., type=float)
        # parser.add_argument('--weight_cls', default=0., type=float)
        # parser.add_argument('--weight_fine', default=0., type=float)
        # parser.add_argument('--weight_channels', default=0., type=float)
        # parser.add_argument('--weight_dsa', default=0., type=float)
        # parser.add_argument('--pos_scale', default=0.6, type=float)
        # parser.add_argument('--infoNCE_logit', default=3.65, type=float)

        # D means 1*1024 feature from Drone-branch S means 1*1024 feature from Satellite-branch
        # D_fine means fine-grained features from Drone-branch and S_fine means fine-grained features from Satellite-branch
        # the loss between Drone and Sat is the traditional infoNCE loss
        # the loss between Drone and Drone or between Sat and Sat is the CAM loss we proposed

        # -- the weights of loss are learnable
        # parser.add_argument('--weight_D_S', default=1.0, type=float)
        # parser.add_argument('--weight_D_D', default=0., type=float)
        # parser.add_argument('--weight_S_S', default=0., type=float)
        # parser.add_argument('--weight_D_fine_S_fine', default=0., type=float)
        # parser.add_argument('--weight_D_fine_D_fine', default=0., type=float)
        # parser.add_argument('--weight_S_fine_S_fine', default=0., type=float)

        # =========================================================================
        parser.add_argument('--blocks_for_PPB', default=3, type=int)

        parser.add_argument('--if_learn_ECE_weights', default=True, type=bool)
        parser.add_argument('--learn_weight_D_D', default=0., type=float)
        parser.add_argument('--learn_weight_S_S', default=0., type=float)
        parser.add_argument('--learn_weight_D_fine_S_fine', default=1.0, type=float)
        parser.add_argument('--learn_weight_D_fine_D_fine', default=0.5, type=float)
        parser.add_argument('--learn_weight_S_fine_S_fine', default=0., type=float)

        parser.add_argument('--if_use_plus_1', default=False, type=bool)
        parser.add_argument('--if_use_multiply_1', default=True, type=bool)
        parser.add_argument('--only_DS', default=False, type=bool)
        parser.add_argument('--only_fine', default=True, type=bool)
        parser.add_argument('--DS_and_fine', default=False, type=bool)

        # --

        # Training Config
        parser.add_argument('--mixed_precision', default=True, type=bool)
        parser.add_argument('--custom_sampling', default=True, type=bool)
        parser.add_argument('--seed', default=1, type=int, help='random seed')
        parser.add_argument('--epochs', default=1, type=int, help='1 epoch for 1652')
        parser.add_argument('--batch_size', default=8, type=int, help='remember the bs is for 2 branches')
        parser.add_argument('--verbose', default=True, type=bool)
        parser.add_argument('--gpu_ids', default=(0, 1, 2, 3), type=tuple)

        # Eval Config
        # parser.add_argument('--batch_size_eval', default=32, type=int)
        # parser.add_argument('--eval_every_n_epoch', default=1, type=int)
        # parser.add_argument('--normalize_features', default=True, type=bool)
        # parser.add_argument('--eval_gallery_n', default=-1, type=int)

        # Optimizer Config
        # parser.add_argument('--clip_grad', default=100.0, type=float)
        # parser.add_argument('--decay_exclue_bias', default=False, type=bool)
        # parser.add_argument('--grad_checkpointing', default=False, type=bool)

        # Loss Config
        # parser.add_argument('--label_smoothing', default=0.1, type=float)

        # Learning Rate Config
        # parser.add_argument('--lr', default=0.001, type=float, help='1 * 10^-4 for ViT | 1 * 10^-1 for CNN')
        # parser.add_argument('--scheduler', default="cosine", type=str, help=r'"polynomial" | "cosine" | "constant" | None')
        # parser.add_argument('--warmup_epochs', default=0.1, type=float)
        # parser.add_argument('--lr_end', default=0.0001, type=float)
        #
        # # Learning part Config
        # parser.add_argument('--lr_mlp', default=None, type=float)
        # parser.add_argument('--lr_decouple', default=None, type=float)
        # parser.add_argument('--lr_blockweights', default=2, type=float)
        # parser.add_argument('--lr_weight_ECE', default=None, type=float)

        # Dataset Config
        # parser.add_argument('--dataset', default='U1652-D2S', type=str, help="'U1652-D2S' | 'U1652-S2D'")
        # parser.add_argument('--data_folder', default='./data/U1652', type=str)
        parser.add_argument('--dataset_name', default='U1652', type=str)

        # Augment Images Config
        parser.add_argument('--prob_flip', default=0.5, type=float, help='flipping the sat image and drone image simultaneously')

        # Savepath for model checkpoints Config
        parser.add_argument('--model_path', default='./checkpoints_xxxx/university', type=str)

        # Eval before training Config
        parser.add_argument('--zero_shot', default=False, type=bool)

        # Checkpoint to start from Config
        parser.add_argument('--checkpoint_start', default=None)

        # Set num_workers to 0 if on Windows Config
        parser.add_argument('--num_workers', default=0 if os.name == 'nt' else 4, type=int)

        # Train on GPU if available Config
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)

        # For better performance Config
        parser.add_argument('--cudnn_benchmark', default=True, type=bool)

        # Make cudnn deterministic Config
        parser.add_argument('--cudnn_deterministic', default=False, type=bool)

        args = parser.parse_args(namespace=self)


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#
config = Configuration()

def get_camp_model(model_name, ck_path, device):
    import warnings
    warnings.filterwarnings('ignore')

    if 'base' in model_name:
        config.model = 'convnext_base.fb_in22k_ft_in1k_384'
    elif 'tiny' in model_name:
        config.model = 'convnext_tiny.in12k_ft_in1k'
    elif 'pico' in model_name:
        config.model = 'convnext_pico.d1_in1k'
    elif 'small' in model_name:
        config.model = 'convnext_small'

    model = make_model(config)
    checkpoint = torch.load(ck_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print("Missing keys:", missing_keys)  # 模型需要但checkpoint缺少的键
    print("Unexpected keys:", unexpected_keys)  # checkpoint存在但模型不需要的键


    return model
