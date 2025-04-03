import os
from time import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
from arguments import parse_args
from dataset import load_dataset
from meter import RunningMeter, BestMeter
from model import Classifier
from torch import optim
from torch.optim.lr_scheduler import StepLR
from utils import save_meter, compute_best_metrics, compute_metrics, \
    model_save_name, set_all_seeds, save_model, update_args

import copy
import argparse

parser = argparse.ArgumentParser(description='???')



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# param_map_naive = {
#     "gpt_v1-1":{'cairo': {'all-distilroberta-v1': {'acc': 0.8928571428571429, 'f1': 0.892975841609479, 'f1_m': 0.8558543331110628, 'best_params': {'dataset': 'cairo', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'aruba': {'all-distilroberta-v1': {'f1': 0.9665610675603775, 'acc': 0.9669921875, 'f1_m': 0.9201025549609653, 'best_params': {'dataset': 'aruba', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'milan': {'all-distilroberta-v1': {'f1': 0.9267403103435521, 'acc': 0.9270833333333334, 'f1_m': 0.8022854177243243, 'best_params': {'dataset': 'milan', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'kyoto7': {'all-distilroberta-v1': {'f1': 0.7648533593539721, 'acc': 0.76953125, 'f1_m': 0.737249815208684, 'best_params': {'dataset': 'kyoto7', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}},},
#     "v1-1":{'cairo': {'all-distilroberta-v1': {'acc': 0.84375, 'f1': 0.8364516304773658, 'f1_m': 0.731014363787473, 'best_params': {'dataset': 'cairo', 'emb': 'v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'aruba': {'all-distilroberta-v1': {'f1': 0.9296445733007359, 'acc': 0.9308894230769231, 'f1_m': 0.8396067455540375, 'best_params': {'dataset': 'aruba', 'emb': 'v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'milan': {'all-distilroberta-v1': {'f1': 0.9093152522290856, 'acc': 0.9131944444444444, 'f1_m': 0.7637782014305293, 'best_params': {'dataset': 'milan', 'emb': 'v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'kyoto7': {'all-distilroberta-v1': {'f1': 0.8033519439848433, 'acc': 0.8125, 'f1_m': 0.7674997178710962, 'best_params': {'dataset': 'kyoto7', 'emb': 'v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}},
#     "v1":{'cairo': {'all-distilroberta-v1': {'acc': 0.8671875, 'f1': 0.8624788457049486, 'f1_m': 0.7763414176859554, 'best_params': {'dataset': 'cairo', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'aruba': {'all-distilroberta-v1': {'f1': 0.9364351345288495, 'acc': 0.9375, 'f1_m': 0.8560801838453047, 'best_params': {'dataset': 'aruba', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'milan': {'all-distilroberta-v1': {'f1': 0.93308569721629, 'acc': 0.9340277777777778, 'f1_m': 0.8125135804840005, 'best_params': {'dataset': 'milan', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'kyoto7': {'all-distilroberta-v1': {'f1': 0.7096799383013593, 'acc': 0.734375, 'f1_m': 0.5958709704733034, 'best_params': {'dataset': 'kyoto7', 'emb': 'v1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}},
#     "gpt_v1":{'cairo': {'all-distilroberta-v1': {'acc': 0.8705357142857143, 'f1': 0.8666782883169556, 'f1_m': 0.8389966009972049, 'best_params': {'dataset': 'cairo', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'aruba': {'all-distilroberta-v1': {'f1': 0.9606014505982688, 'acc': 0.961328125, 'f1_m': 0.910719891007556, 'best_params': {'dataset': 'aruba', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'milan': {'all-distilroberta-v1': {'f1': 0.9216186562715394, 'acc': 0.921875, 'f1_m': 0.7892121855752917, 'best_params': {'dataset': 'milan', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}, 'kyoto7': {'all-distilroberta-v1': {'f1': 0.7354429025441687, 'acc': 0.74609375, 'f1_m': 0.6871466555140024, 'best_params': {'dataset': 'kyoto7', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}}}}
    
# }

# param_map_naive={
#     "gpt_v1":{'cairo': {'sentence-t5-base': {'acc': 0.9084821428571429, 'f1': 0.9074282950529972, 'f1_m': 0.8577943502426724, 'best_params': {'dataset': 'cairo', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'aruba': {'sentence-t5-base': {'f1': 0.959908189780252, 'acc': 0.961328125, 'f1_m': 0.8992822216486372, 'best_params': {'dataset': 'aruba', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'milan': {'sentence-t5-base': {'f1': 0.91524651985864, 'acc': 0.9155092592592593, 'f1_m': 0.8323450412150877, 'best_params': {'dataset': 'milan', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.001', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'kyoto7': {'sentence-t5-base': {'f1': 0.7629692578659092, 'acc': 0.76953125, 'f1_m': 0.7305324768391522, 'best_params': {'dataset': 'kyoto7', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0001', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}},
#     "gpt_v1-1":{'cairo': {'sentence-t5-base': {'acc': 0.9107142857142857, 'f1': 0.9094831838035107, 'f1_m': 0.8440387719743935, 'best_params': {'dataset': 'cairo', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'aruba': {'sentence-t5-base': {'f1': 0.9667049845800191, 'acc': 0.9677734375, 'f1_m': 0.9187743040314847, 'best_params': {'dataset': 'aruba', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'milan': {'sentence-t5-base': {'f1': 0.9272388945750285, 'acc': 0.9276620370370371, 'f1_m': 0.8390464801184363, 'best_params': {'dataset': 'milan', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0001', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'kyoto7': {'sentence-t5-base': {'f1': 0.77376066631427, 'acc': 0.77734375, 'f1_m': 0.7569580206842976, 'best_params': {'dataset': 'kyoto7', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0001', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}},
#     "v1":{'cairo': {'sentence-t5-base': {'acc': 0.8671875, 'f1': 0.8617910132650919, 'f1_m': 0.8098861745282339, 'best_params': {'dataset': 'cairo', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'aruba': {'sentence-t5-base': {'f1': 0.9612019653959146, 'acc': 0.9621394230769231, 'f1_m': 0.9027148109119649, 'best_params': {'dataset': 'aruba', 'emb': 'v1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0001', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'milan': {'sentence-t5-base': {'f1': 0.9114597732561273, 'acc': 0.9149305555555556, 'f1_m': 0.7650780445542025, 'best_params': {'dataset': 'milan', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'kyoto7': {'sentence-t5-base': {'f1': 0.6757239145658263, 'acc': 0.71875, 'f1_m': 0.6363403035632859, 'best_params': {'dataset': 'kyoto7', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}},
#     "v1-1":{'cairo': {'sentence-t5-base': {'acc': 0.8515625, 'f1': 0.8458114495798319, 'f1_m': 0.7760765844799458, 'best_params': {'dataset': 'cairo', 'emb': 'v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'aruba': {'sentence-t5-base': {'f1': 0.96253325374563, 'acc': 0.9639423076923077, 'f1_m': 0.910431508816122, 'best_params': {'dataset': 'aruba', 'emb': 'v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'milan': {'sentence-t5-base': {'f1': 0.8994922456699295, 'acc': 0.9045138888888888, 'f1_m': 0.7568766050048938, 'best_params': {'dataset': 'milan', 'emb': 'v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0001', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}, 'kyoto7': {'sentence-t5-base': {'f1': 0.7222569776238255, 'acc': 0.734375, 'f1_m': 0.6169294473642299, 'best_params': {'dataset': 'kyoto7', 'emb': 'v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}}}
# }

param_map_naive = {
    "gpt_v1-1": {
        'cairo': {
            'all-distilroberta-v1': {'acc': 0.8928571428571429, 'f1': 0.892975841609479, 'f1_m': 0.8558543331110628, 'best_params': {'dataset': 'cairo', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'acc': 0.9107142857142857, 'f1': 0.9094831838035107, 'f1_m': 0.8440387719743935, 'best_params': {'dataset': 'cairo', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'aruba': {
            'all-distilroberta-v1': {'f1': 0.9665610675603775, 'acc': 0.9669921875, 'f1_m': 0.9201025549609653, 'best_params': {'dataset': 'aruba', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.9667049845800191, 'acc': 0.9677734375, 'f1_m': 0.9187743040314847, 'best_params': {'dataset': 'aruba', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'milan': {
            'all-distilroberta-v1': {'f1': 0.9267403103435521, 'acc': 0.9270833333333334, 'f1_m': 0.8022854177243243, 'best_params': {'dataset': 'milan', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.9272388945750285, 'acc': 0.9276620370370371, 'f1_m': 0.8390464801184363, 'best_params': {'dataset': 'milan', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0001', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'kyoto7': {
            'all-distilroberta-v1': {'f1': 0.7648533593539721, 'acc': 0.76953125, 'f1_m': 0.737249815208684, 'best_params': {'dataset': 'kyoto7', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.77376066631427, 'acc': 0.77734375, 'f1_m': 0.7569580206842976, 'best_params': {'dataset': 'kyoto7', 'emb': 'gpt_v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0001', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        }
    },
    "v1-1": {
        'cairo': {
            'all-distilroberta-v1': {'acc': 0.84375, 'f1': 0.8364516304773658, 'f1_m': 0.731014363787473, 'best_params': {'dataset': 'cairo', 'emb': 'v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'acc': 0.8515625, 'f1': 0.8458114495798319, 'f1_m': 0.7760765844799458, 'best_params': {'dataset': 'cairo', 'emb': 'v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'aruba': {
            'all-distilroberta-v1': {'f1': 0.9296445733007359, 'acc': 0.9308894230769231, 'f1_m': 0.8396067455540375, 'best_params': {'dataset': 'aruba', 'emb': 'v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.96253325374563, 'acc': 0.9639423076923077, 'f1_m': 0.910431508816122, 'best_params': {'dataset': 'aruba', 'emb': 'v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'milan': {
            'all-distilroberta-v1': {'f1': 0.9093152522290856, 'acc': 0.9131944444444444, 'f1_m': 0.7637782014305293, 'best_params': {'dataset': 'milan', 'emb': 'v1-1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.8994922456699295, 'acc': 0.9045138888888888, 'f1_m': 0.7568766050048938, 'best_params': {'dataset': 'milan', 'emb': 'v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0001', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'kyoto7': {
            'all-distilroberta-v1': {'f1': 0.8033519439848433, 'acc': 0.8125, 'f1_m': 0.7674997178710962, 'best_params': {'dataset': 'kyoto7', 'emb': 'v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.7222569776238255, 'acc': 0.734375, 'f1_m': 0.6169294473642299, 'best_params': {'dataset': 'kyoto7', 'emb': 'v1-1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        }
    },
    "v1": {
        'cairo': {
            'all-distilroberta-v1': {'acc': 0.8671875, 'f1': 0.8624788457049486, 'f1_m': 0.7763414176859554, 'best_params': {'dataset': 'cairo', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'acc': 0.8671875, 'f1': 0.8617910132650919, 'f1_m': 0.8098861745282339, 'best_params': {'dataset': 'cairo', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'aruba': {
            'all-distilroberta-v1': {'f1': 0.9364351345288495, 'acc': 0.9375, 'f1_m': 0.8560801838453047, 'best_params': {'dataset': 'aruba', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.9612019653959146, 'acc': 0.9621394230769231, 'f1_m': 0.9027148109119649, 'best_params': {'dataset': 'aruba', 'emb': 'v1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0001', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'milan': {
            'all-distilroberta-v1': {'f1': 0.93308569721629, 'acc': 0.9340277777777778, 'f1_m': 0.8125135804840005, 'best_params': {'dataset': 'milan', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.9114597732561273, 'acc': 0.9149305555555556, 'f1_m': 0.7650780445542025, 'best_params': {'dataset': 'milan', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'kyoto7': {
            'all-distilroberta-v1': {'f1': 0.7096799383013593, 'acc': 0.734375, 'f1_m': 0.5958709704733034, 'best_params': {'dataset': 'kyoto7', 'emb': 'v1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.6757239145658263, 'acc': 0.71875, 'f1_m': 0.6363403035632859, 'best_params': {'dataset': 'kyoto7', 'emb': 'v1', 'fold': '1', 'lr': '0.001', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        }
    },
    "gpt_v1": {
        'cairo': {
            'all-distilroberta-v1': {'acc': 0.8705357142857143, 'f1': 0.8666782883169556, 'f1_m': 0.8389966009972049, 'best_params': {'dataset': 'cairo', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'acc': 0.9084821428571429, 'f1': 0.9074282950529972, 'f1_m': 0.8577943502426724, 'best_params': {'dataset': 'cairo', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'aruba': {
            'all-distilroberta-v1': {'f1': 0.9606014505982688, 'acc': 0.961328125, 'f1_m': 0.910719891007556, 'best_params': {'dataset': 'aruba', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.959908189780252, 'acc': 0.961328125, 'f1_m': 0.8992822216486372, 'best_params': {'dataset': 'aruba', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.0005', 'wd': '1e-05', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'milan': {
            'all-distilroberta-v1': {'f1': 0.9216186562715394, 'acc': 0.921875, 'f1_m': 0.7892121855752917, 'best_params': {'dataset': 'milan', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.91524651985864, 'acc': 0.9155092592592593, 'f1_m': 0.8323450412150877, 'best_params': {'dataset': 'milan', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.001', 'wd': '0.0', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        },
        'kyoto7': {
            'all-distilroberta-v1': {'f1': 0.7354429025441687, 'acc': 0.74609375, 'f1_m': 0.6871466555140024, 'best_params': {'dataset': 'kyoto7', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.001', 'wd': '0', 'batch_size': '64', 'encoder_name': 'all-distilroberta-v1'}},
            'sentence-t5-base': {'f1': 0.7629692578659092, 'acc': 0.76953125, 'f1_m': 0.7305324768391522, 'best_params': {'dataset': 'kyoto7', 'emb': 'gpt_v1', 'fold': '1', 'lr': '0.0005', 'wd': '0.0001', 'batch_size': '64', 'encoder_name': 'sentence-t5-base'}}
        }
    }
}


def evaluate_with_classifier(config, args=None):
    print('Inside evaluate classifier function')


    # Adding the config params back into the arg parser
    args = update_args(config, args)

    # print(args)



    # Getting the trained model name
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # if args.saved_model_folder is not None:
    #     args.saved_model = os.path.join(
    #         dir_path,
    #         'final_models',
    #         args.saved_model_folder,
    #         model_save_name(args, capture=True) + '.pkl'
    #     )
    # else:
    #     args.saved_model = None


    print("Updated lr and wd")
    embd = param_map_naive.get(args.embedding_type).get(args.dataset)
    if embd:
        args.classifier_lr = float(embd[args.sentence_encoder_name]['best_params']['lr'])
        args.classifier_wd = float(embd[args.sentence_encoder_name]['best_params']['wd'])
    else:
        args.classifier_lr = 0.001
        args.classifier_wd = 0.001

        print(f"No param_map_naive mappings found for {args.dataset}, using defaults: \n"
              f"    args.classifier_lr = {args.classifier_lr} \n"
              f"    args.classifier_wd = {args.classifier_wd}     \n")
        

      
        
    # Load the target data
    data_loaders, dataset_sizes = load_dataset(args, classifier=True)
    
    print(dataset_sizes)

    # Tracking meters
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()

    # Creating the model
    model = Classifier(args).to(args.device)

    # Loading pre-trained weights if available
    # if args.saved_model is not None:
    #     model.load_pretrained_weights(args)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=args.classifier_lr,
                           weight_decay=args.classifier_wd)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.CrossEntropyLoss()


    max_f1 =0.0
    best_model_wts = model.state_dict()
    best_epoch =0
    for epoch in range(0, args.num_epochs):
        since = time()
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        # Training
        model, optimizer, scheduler = train(model,
                                            data_loaders["train"],
                                            criterion,
                                            optimizer,
                                            scheduler,
                                            args,
                                            epoch,
                                            dataset_sizes["train"],
                                            running_meter)

        # Validation
        evaluate(model,
                 data_loaders["val"],
                 args,
                 criterion,
                 epoch,
                 phase="val",
                 dataset_size=dataset_sizes["val"],
                 running_meter=running_meter)

        # Evaluating on the test data
        evaluate(model,
                 data_loaders["test"],
                 args,
                 criterion,
                 epoch,
                 phase="test",
                 dataset_size=dataset_sizes["test"],
                 running_meter=running_meter)
        
      
        if max_f1 < max(running_meter.f1_score_weighted['val']):
            max_f1 = max(running_meter.f1_score_weighted['val'])
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            
            
        # Printing the time taken
        time_elapsed = time() - since
        print('Epoch {} completed in {:.0f}m {:.0f}s'
              .format(epoch, time_elapsed // 60, time_elapsed % 60))

  
    best_meter = compute_best_metrics(running_meter, best_meter,
                                      classifier=True)
    running_meter.update_best_meter(best_meter)
    print("saving meter")
    save_meter(args, running_meter, classifier=True)
    print('Best weights at epoch: {}'.
          format(running_meter.best_meter.epoch))
    
    
    best_meter.display()
    

    model.load_state_dict(best_model_wts)
    
    save_model(model, args, best_epoch)


    return


def train(model, data_loader, criterion, optimizer, scheduler, args, epoch,
          dataset_size, running_meter):
    # Setting the model to training mode
    model.train()

    # Set only softmax layer to trainable
    if args.learning_schedule == 'last_layer':
        model.freeze_encoder_layers()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []


    # Iterating over the data
    for inputs, labels in data_loader:
        inputs = inputs.float().to(args.device)
        labels = labels.long().to(args.device)

        optimizer.zero_grad()

        ##### Encode your inputs here #####


        ###################################

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        actual_labels.extend(labels.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())

    scheduler.step()

    # Statistics
    loss = running_loss / dataset_size
    _ = compute_metrics(actual_labels, pred_labels,
                        'train', running_meter, loss,
                        epoch)

    return model, optimizer, scheduler


def evaluate(model, data_loader, args, criterion, epoch, phase, dataset_size,
             running_meter):
    # Setting the model to eval mode
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []

    # Iterating over the data
    for inputs, labels in data_loader:
        inputs = inputs.float().to(args.device)
        labels = labels.long().to(args.device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        actual_labels.extend(labels.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())

    # Statistics
    loss = running_loss / dataset_size
    _ = compute_metrics(actual_labels, pred_labels,
                        phase, running_meter, loss,
                        epoch)

    return


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    parser.add_argument('--fold',        type=int,default=2)
    parser.add_argument('--input_size',  type=int,default=768)
    parser.add_argument('--dataset', type=str, help='dataset to be evaluated.',default='aruba')
    parser.add_argument('--root_dir', type=str, help='absolute path where stuff will be saved.',default='/coc/pcba1/mthukral3/gt/TDOST/folds/pre-segmented/')
    parser.add_argument('--num_classes', type=int, help='number of classes in the dataset', default=9)
    parser.add_argument('--num_epochs', type=int, help='number of epochs to train', default=75)
    args_here = parser.parse_args()


    args = parse_args(
        fold=args_here.fold,
        input_size=args_here.input_size,
        root_dir=args_here.root_dir, 
        dataset=args_here.dataset, 
        num_classes=args_here.num_classes, 
        num_epochs=args_here.num_epochs, 
        )
    
    set_all_seeds(args.random_seed)
    print(args)

    evaluate_with_classifier(config={}, args=args)

    print('------ Evaluation complete! ------')
