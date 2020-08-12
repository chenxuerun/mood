import os
import sys
import random
from math import ceil

import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm

from .nifti_io import ni_save, ni_load
from .constant import *


def init_parameter(module):
    if hasattr(module, 'weight'):
        nn.init.constant_(module.weight, 1e-2)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 1e-2)


# 将origin_dict中的值换成map中的值。
def transform_dict_data(origin_dict, map):
    map_keys = map.keys()
    for key, value in origin_dict.items():
        if value in list(map_keys):
            origin_dict[key] = map[value]


def save_images(pred_dir, f_name, ni_aff, score=None, ori=None, rec=None):
    if score is not None:
        score_dir = os.path.join(pred_dir, 'score')
        if not os.path.exists(score_dir): os.mkdir(score_dir)
        ni_save(os.path.join(score_dir, f_name), score, ni_aff)

    if ori is not None:
        ori_dir = os.path.join(pred_dir, 'ori')
        if not os.path.exists(ori_dir): os.mkdir(ori_dir)
        ni_save(os.path.join(ori_dir, f_name), ori, ni_aff)

    if rec is not None:
        rec_dir = os.path.join(pred_dir, 'rec')
        if not os.path.exists(rec_dir): os.mkdir(rec_dir)
        ni_save(os.path.join(rec_dir, f_name), rec, ni_aff)


def clip_image(input_folder, output_folder):
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):
            if not os.path.exists(os.path.join(output_folder, folder_name)):
                os.mkdir(os.path.join(output_folder, folder_name))
            for f_name in os.listdir(folder_path):
                ni_file = os.path.join(folder_path, f_name)
                ni_data, ni_affine = ni_load(ni_file)
                ni_data = np.clip(ni_data, a_max=1.0, a_min=0.0)
                ni_save(os.path.join(output_folder, folder_name, f_name), ni_data, ni_affine)


# def train_model(model, opt, data, batch_idx):
#     if RECIPE == 'predict':
#         input = data[:, range(SEE_SLICE), :, :].to(DEVICE)
#         label = data[:, [SEE_SLICE], :, :].to(DEVICE)
#         net_out = model(input)
#         loss = torch.mean(torch.pow(net_out - label, 2))
    # elif RECIPE == 'mask':
    #     label = data.to(DEVICE)
    #     if random.random() < 0.8:
    #         if FIXED_MASK_SQUARE_SIZE: square_size = SQUARE_SIZE
    #         else: square_size = (0, data.shape[2] // 2)
    #         ce_tensor = get_square_mask(data.shape, square_size=square_size, n_squares=1,
    #             noise_val=(torch.min(data).item(), torch.max(data).item()))
    #         ce_tensor = torch.from_numpy(ce_tensor).float()
    #         inpt_noisy = torch.where(ce_tensor != 0, ce_tensor, data)
    #         input = inpt_noisy.to(DEVICE)
    #     else:
    #         input = label
    #     net_out = model(input)
    #     loss = torch.mean(torch.pow(net_out - label, 2))
#     elif RECIPE == 'rotate':
#         if BLOCK:
#             label = data.to(DEVICE)
#             inpt_rot = torch.rot90(data, 1, [2, 3])
#             input = inpt_rot.to(DEVICE)
#             net_out = model(input)
#             loss = torch.mean(torch.pow(net_out - label, 2))
#         else:
#             a, b = data.chunk(2, 2)
#             a1, a2 = a.chunk(2, 3)
#             b1, b2 = b.chunk(2, 3)
#             losses = []
#             for inpt_split in list([a1, a2, b1, b2]):
#                 inpt_rot = torch.rot90(inpt_split, 1, [2, 3])
#                 inpt_rec = self.model(inpt_rot)
#                 loss = torch.pow(inpt_split - inpt_rec, 2)
#                 losses.append(loss)
#             losses = torch.stack(losses)
#             loss = torch.mean(losses)
#     else:
#         label = data.to(DEVICE)
#         input = label
#         net_out = model(input)
#         loss = torch.mean(torch.pow(net_out - label, 2))

#     opt.zero_grad()
#     loss.backward()
#     opt.step()

#     return loss, input, net_out


def init_validation_dir(algo_name, dataset_dir):
    eval_dir = os.path.join(dataset_dir, 'eval')
    if not os.path.exists(eval_dir):    os.mkdir(eval_dir)
    algo_dir = os.path.join(eval_dir, algo_name)
    if not os.path.exists(algo_dir):    os.mkdir(algo_dir)
    pred_dir = os.path.join(algo_dir, 'predict')
    if not os.path.exists(pred_dir):    os.mkdir(pred_dir)
    score_dir = os.path.join(algo_dir, 'score')
    if not os.path.exists(score_dir):   os.mkdir(score_dir)

    pred_pixel_dir = os.path.join(pred_dir, 'pixel')
    if not os.path.exists(pred_pixel_dir):  os.mkdir(pred_pixel_dir)
    pred_sample_dir = os.path.join(pred_dir, 'sample')
    if not os.path.exists(pred_sample_dir): os.mkdir(pred_sample_dir)

    return score_dir, pred_pixel_dir, pred_sample_dir