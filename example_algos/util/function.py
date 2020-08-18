import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import random
from math import ceil
import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from util.nifti_io import ni_save, ni_load
from util.constant import *


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
    ni_aff = ni_aff.astype(np.float64)
    if score is not None:
        score_dir = os.path.join(pred_dir, 'score')
        if not os.path.exists(score_dir): os.mkdir(score_dir)
        score = score.astype(np.float64)
        ni_save(os.path.join(score_dir, f_name), score, ni_aff)

    if ori is not None:
        ori_dir = os.path.join(pred_dir, 'ori')
        if not os.path.exists(ori_dir): os.mkdir(ori_dir)
        ori = ori.astype(np.float64)
        ni_save(os.path.join(ori_dir, f_name), ori, ni_aff)

    if rec is not None:
        rec_dir = os.path.join(pred_dir, 'rec')
        if not os.path.exists(rec_dir): os.mkdir(rec_dir)
        rec = rec.astype(np.float64)
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


def template_statistics(test_dir):
    import matplotlib.pyplot as plt

    predict_dir = os.path.join(test_dir, 'eval', 'temmat', 'predict')
    assert os.path.exists(predict_dir), '先预测，再统计'
    statistics_dir = os.path.join(predict_dir, 'statistics')
    if not os.path.exists(statistics_dir):
        os.mkdir(statistics_dir)

    handle = tqdm(enumerate(os.listdir(os.path.join(predict_dir, 'pixel', 'rec'))))
    for i, file_name in handle:
        prefix = file_name.split('.')[0]
        each_statistics_dir = os.path.join(statistics_dir, prefix)
        if not os.path.exists(each_statistics_dir): os.mkdir(each_statistics_dir)

        score, ni_aff = ni_load(os.path.join(predict_dir, 'pixel', 'score', file_name))
        flatten_score = score.flatten()

        # 整体打分直方图
        plt.hist(flatten_score, bins=50, log=False)
        plt.savefig(os.path.join(each_statistics_dir, 'whole_score_histogram'))
        plt.cla()

        with open(os.path.join(test_dir, 'label', 'sample', file_name + '.txt'), "r") as f:
            sample_label = f.readline()
        sample_label = int(sample_label)

        if sample_label == 1:
            # 异常区域打分直方图
            label, _ = ni_load(os.path.join(test_dir, 'label', 'pixel', file_name))
            abnormal_area_score = score[label == 1]
            plt.hist(abnormal_area_score, bins=50, log=False)
            plt.savefig(os.path.join(each_statistics_dir, 'abnormal_area_score_histogram'))
            plt.cla()

            abnormal_number = len(abnormal_area_score)
            # print(f'abnormal_number: {abnormal_number}')
        elif sample_label == 0:
            abnormal_number = 10000
        else: raise Exception(f'sample_label有问题: {sample_label}')

        # 高分区域打分直方图
        ordered_flatten_score = np.sort(flatten_score)[::-1]
        large_score = ordered_flatten_score[0: abnormal_number]
        plt.hist(large_score, bins=50, log=False)
        plt.savefig(os.path.join(each_statistics_dir, 'max_score_area_score_histogram'))
        plt.cla()

        max_score = large_score[0]
        img = score / max_score
        ni_save(os.path.join(each_statistics_dir, 'normalized'), img, ni_aff)


def template_match_ex(test_dir): # 读进来的是nii.gz
    from util.configure import TRAIN_DATASET_DIR
    from scripts.evalresults import eval_dir

    score_dir, pred_pixel_dir, pred_sample_dir = init_validation_dir('temmat', test_dir)
    templates = load_array(os.path.join(TRAIN_DATASET_DIR, 'preprocessed'))

    print('predict')
    for f_name in os.listdir(os.path.join(test_dir, 'data')):
        print(f'f_name: {f_name}')
        np_array, ni_aff = ni_load(os.path.join(test_dir, 'data', f_name))

        score, rec = template_match(templates, np_array)
        save_images(pred_pixel_dir, f_name, ni_aff, score=score, rec=rec)

        sample_score = get_sample_score(score)
        with open(os.path.join(pred_sample_dir, f_name + ".txt"), "w") as target_file:
            target_file.write(str(sample_score))
    
    eval_dir(pred_dir=os.path.join(pred_pixel_dir, 'score'), label_dir=os.path.join(test_dir, 'label', 'pixel'), mode='pixel', save_file=os.path.join(score_dir, 'pixel'))
    eval_dir(pred_dir=pred_sample_dir, label_dir=os.path.join(test_dir, 'label', 'sample'), mode='sample', save_file=os.path.join(score_dir, 'sample'))


def template_match(imgs, np_array): # imgs 四维
    min_score_num = np.inf
    min_score_index = -1

    length = len(imgs)
    handle = tqdm(enumerate(imgs))
    for i, img in handle:
        score = (img - np_array) ** 2
        score_num = np.sum(score)
        if score_num < min_score_num:
            min_score_num = score_num
            min_score_index = i

        handle.set_description_str(f'{i+1}/{length}')

    rec = imgs[min_score_index]
    score = (rec - np_array) ** 2
    return score, rec


def get_sample_score(score):
    slice_scores = []
    for sli in score:
        slice_score = np.mean(sli)
        slice_scores.append(slice_score)
    return np.max(slice_scores)


def load_array(path):
    print(f'load_array')
    imgs = []
    handle = tqdm(os.listdir(path))
    for fname in handle:
        if fname.endswith('data.npy'):
            np_array = np.load(os.path.join(path, fname))
            imgs.append(np_array)
    return imgs


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


if __name__ == '__main__':
    from util.configure import TEST_DATASET_DIR
    # template_match_ex(test_dir=TEST_DATASET_DIR)
    template_statistics(test_dir=TEST_DATASET_DIR)