import os
import sys

from .constant import *


TRAIN_DATASET_DIR = '/home/cxr/Program_Datas/mood/brain_train'
TEST_DATASET_NAME = 'brain_toy'
TEST_DATASET_DIR = os.path.join('/home/cxr/Program_Datas/mood', TEST_DATASET_NAME)
# TEST_DATASET_DIR = '/home/cxr/toy'
LOAD_MODEL_NAME = 'unet_origin_resolution'

# keywords
BASIC_KWS = {
    'logger': 'tensorboard',
    'log_dir': os.path.join(TRAIN_DATASET_DIR, 'log'),
    'train_data_dir': os.path.join(TRAIN_DATASET_DIR, 'preprocessed'),
    'test_dir': TEST_DATASET_DIR,

    'load': True,
    'load_path': os.path.join(TRAIN_DATASET_DIR, 'log', LOAD_MODEL_NAME, 'checkpoint'),
    'name': LOAD_MODEL_NAME,
}

# 只包含了公共的参数，不同recipe对应的参数要手动添加
TRAIN_KWS = {
    'print_every_iter': 100,
    'n_epochs': 10,
    'lr': 1e-4,
    'batch_size': 16,
    'origin_size': 256,
    'select_dim': 0,
}

OTHER_KWS = {
    'see_slice': 5,
    'data_augment_prob': 0.8,
    'mask_square_size': 15,
    'res_size': 64,
    'minus_low': False,
    'loss_type': '7loss',
}

CONFIGURE_DICT = {
    'zunet': {},
    'zcae': {},

    'unet': {
        'base_num_features': 16,
        'num_classes': 1,
        'num_pool': 5,
        'num_conv_per_stage': 2,
        'feat_map_mul_on_downscale': 2,
        'conv_op': 'conv2d',
        'norm_op': 'batchnorm2d',
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True, 'momentum': 0.1},
        'dropout_op': 'dropout2d',
        'dropout_op_kwargs': {'p': 0.5, 'inplace': True},
        'nonlin': 'leakyrelu',
        'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True},
        'deep_supervision': False,
        'dropout_in_localization': False,
        'final_nonlin': 'sigmoid',
        'weightInitializer': 'init_parameter',
        'pool_op_kernel_sizes': None,
        'conv_kernel_sizes': None,
        'upscale_logits': False,
        'convolutional_pooling': True,
        'convolutional_upsampling': True,
        'max_num_features': None,
        'seg_output_use_bias': False,
    },
}