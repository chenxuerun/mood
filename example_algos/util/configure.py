import os
import sys

from .constant import *
from example_algos.nnunet.utilities.nd_softmax import softmax_helper
from .function import init_parameter
from example_algos.nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin

TRAIN_DATASET_DIR = '/home/cxr/Program_Datas/mood/brain_train'
TEST_DATASET_NAME = 'validate_whole'
TEST_DATASET_DIR = os.path.join('/home/cxr/Program_Datas/mood', TEST_DATASET_NAME)
# TEST_DATASET_DIR = '/home/cxr/toy'
LOAD_MODEL_NAME = 'unet_mask'


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
    'n_epochs': 20,
    'lr': 1e-4,
    'batch_size': 16,
    'target_size': 128,
    'select_dim': 0,
}

OTHER_KWS = {
    'see_slice': 5,
    'data_augment_prob': 0.8,
    'mask_square_size': 15,
}

CONFIGURE_DICT = {
    'zunet': {},
    'zcae': {},

    'ae2d': {
        'z_dim': 128,
        'fmap_sizes': (16, 64, 256, 1024),
        'temp_sizes': (64, 32,   16,         8),
    },

    'ae2d-722':{
        'lr': 5e-5,
        'print_every_iter': 100,
        'target_size': 256,
        'batch_size': 16,
        'z_dim': 2048,
        'fmap_sizes': (   16, 32, 64, 128, 256, 512),
        'temp_sizes': (128, 64, 32,   16,       8,      4),
        'other_keywords': ('clip', 'fc')
    },

    'ae2d-test':{
        'lr': 1e-4,
        'print_every_iter': 100,
        'target_size': 256,
        'batch_size': 16,
        'z_dim': 1024,
        'fmap_sizes': (   16, 32, 64, 128, 256, 512),
        'temp_sizes': (128, 64, 32,   16,       8,      4),
        # 'other_keywords': ('sigmoid', 'batch_norm' , 'fc')
        # 'other_keywords': ('clip', 'batch_norm')
        'other_keywords': ('clip', 'batch_norm', 'fc')
    },

    'unet': {
        'base_num_features': 16,
        'num_classes': 1,
        'num_pool': 5,
        'num_conv_per_stage': 1,
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

    'unet3d-sigmoid': {
        'lr': 2e-5,
        'print_every_iter': 10,
        'target_size': 128,
        'batch_size': 4,
        'z_dim': 128,
        'fmap_sizes': (16, 64, 256, 1024),
        'temp_sizes': (64, 32,   16,         8),
    },

    'ae3d': {
        'lr': 1e-4,
        'print_every_iter': 10,
        'target_size': 128,
        'batch_size': 4,
        'z_dim': 128,
        'fmap_sizes': (16, 64, 256, 1024),
        'temp_sizes': (64, 32,   16,         8),
    },

    'ae3d-719':{
        'lr': 1e-5,
        'print_every_iter': 10,
        'target_size': 256,
        'batch_size': 4,
        'z_dim': 128,
        'fmap_sizes': (   16, 64, 256, 1024, 1024),
        'temp_sizes': (128, 64,    32,   16,         8),
        'memory_consumption': 10821
    },

    'ae3d-715':{
        'lr': 1e-4,
        'print_every_iter': 10,
        'target_size': 256,
        'batch_size': 1,
        'z_dim': 1024,
        'fmap_sizes': (     8, 24, 72, 144, 288),
        'temp_sizes': (128, 64, 32,   16,      8),
        'memory_consumption': 7791
    },
}