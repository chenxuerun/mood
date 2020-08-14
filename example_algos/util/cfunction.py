import os

import torch
import numpy as np

from .constant import DEVICE


class FuncFactory:

    def getFunction(self, fn_name, *args):
        if hasattr(self, fn_name):
            return getattr(self, fn_name)
        else:
            create_fn_name = 'create_' + fn_name + '_fn'
            if not hasattr(self, create_fn_name):
                raise Exception(f'工厂无法创造函数：{fn_name}')
            return getattr(self, create_fn_name)(*args)


    # 加一些训练参数
    def create_modify_train_kws_fn(self, recipe, kws):
        if recipe == 'predict':
            def modify_train_kws(train_kws):
                train_kws['see_slice'] = kws['see_slice']
        elif recipe == 'mask':
            def modify_train_kws(train_kws):
                train_kws['data_augment_prob'] = kws['data_augment_prob']
                train_kws['mask_square_size'] = kws['mask_square_size']
        else:
            def modify_train_kws(train_kws): pass
        return modify_train_kws


    # 根据训练参数的情况，修改模型参数
    def create_modify_model_kws_fn(self, train_kws):
        model_type = train_kws['model_type']
        recipe = train_kws['recipe']

        func_list = []

        if recipe == 'predict':
            def modify_model_kws_fn1(model_kws):
                model_kws['in_channels'] = train_kws['see_slice']
        else:
            def modify_model_kws_fn1(model_kws):
                model_kws['in_channels'] = 1
        func_list.append(modify_model_kws_fn1)
        
        if model_type == 'unet':
            def modify_model_kws_fn2(model_kws):
                from .function import transform_dict_data, init_parameter
                from example_algos.nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin
                from example_algos.nnunet.utilities.nd_softmax import softmax_helper
                UNET_OBJ_MAP = {
                    'conv2d': torch.nn.Conv2d,
                    'batchnorm2d': torch.nn.BatchNorm2d,
                    'dropout2d': torch.nn.Dropout2d,
                    'leakyrelu': torch.nn.LeakyReLU,
                    'softmax_helper': softmax_helper,
                    'sigmoid': torch.sigmoid,
                    'init_parameter': init_parameter,
                    'net_block': ConvDropoutNormNonlin,
                }
                transform_dict_data(model_kws, UNET_OBJ_MAP)
        else:
            def modify_model_kws_fn2(model_kws): pass
        func_list.append(modify_model_kws_fn2)

        def modify_model_kws(model_kws):
            for func in func_list:
                func(model_kws)
        
        return modify_model_kws


    def create_run_fn(self, run_mode):
        if run_mode == 'train':
            def run(algo):  algo.train()
        elif run_mode == 'predict':
            def run(algo):  algo.predict()
        elif run_mode == 'validate':
            def run(algo):  algo.validate()
        elif run_mode == 'statistics':
            def run(algo):  algo.statistics()
        else:
            raise Exception(f'未知run_mode：{run_mode}')
        return run


    def create_get_slices_fn(self, train_kws):
        if train_kws['recipe'] == 'predict':
            offset = train_kws['see_slice']
        else: offset = 0
        def get_slices(i, npy_file, file_len):
            slices = [(i, npy_file, j) for j in range(0, file_len - offset)]
            return slices
        return get_slices


    def create_get_data_slice_num_fn(self, train_kws):
        if train_kws['recipe'] == 'predict':
            slice_num = train_kws['see_slice'] + 1
        else: slice_num = 1
        def get_data_slice_num(): return slice_num
        return get_data_slice_num


    def create_get_slice_data_fn(self, train_kws):
        select_dim = train_kws['select_dim']
        if select_dim == 0:
            def get_slice_data(numpy_array, slice_idx, slice_num):
                return numpy_array[slice_idx : slice_idx + slice_num, :, :]
        elif select_dim == 1:
            def get_slice_data(numpy_array, slice_idx, slice_num):
                return numpy_array[:, slice_idx : slice_idx + slice_num, :].transpose(1, 0, 2)
        elif select_dim == 2:
            def get_slice_data(numpy_array, slice_idx, slice_num):
                return numpy_array[:, :, slice_idx : slice_idx + slice_num].transpose(2, 0, 1)
        else: raise Exception(f'select_dim取值有误：{select_dim}')
        return get_slice_data


    def create_calculate_loss_fn(self, train_kws):
        def calculate_loss(model, input, label):
            out = model(input)
            loss = torch.mean(torch.pow(out - label, 2))
            return loss
        return calculate_loss


    # data 是一个batch的数据
    def create_get_input_label_fn(self, train_kws):
        recipe = train_kws['recipe']

        if recipe == 'predict':
            def get_input_label(data):
                input = data[:, range(train_kws['see_slice']), :, :].to(DEVICE)
                label = data[:, [train_kws['see_slice']], :, :].to(DEVICE)
                return input, label
        elif recipe == 'mask':
            import random
            from .ce_noise import get_square_mask
            def get_input_label(data):
                label = data.to(DEVICE)
                if random.random() < train_kws['data_augment_prob']:
                    ce_tensor = get_square_mask(data.shape, square_size=train_kws['mask_square_size'], 
                        n_squares=1, noise_val=(torch.min(data).item(), torch.max(data).item()), data=data)
                    ce_tensor = torch.from_numpy(ce_tensor).float()
                    input_noisy = torch.where(ce_tensor != 0, ce_tensor, data)
                    input = input_noisy.to(DEVICE)
                else:
                    input = label
                return input, label
        elif recipe == 'rotate':
            def get_input_label(data):
                label = data.to(DEVICE)
                input = torch.rot90(label, 1, [2, 3])
                return input, label
        elif recipe == 'split_rotate':
            def get_input_label(data):
                a, b = data.chunk(2, 2)
                a1, a2 = a.chunk(2, 3)
                b1, b2 = b.chunk(2, 3)
                label = torch.cat((a1, a2, b1, b2), 0).to(DEVICE)
                input = torch.rot90(label, 1, [2, 3])
                return input, label
        else:
            def get_input_label(data):
                input = data.to(DEVICE)
                label = input
                return input, label

        return get_input_label

    
    def create_transpose_fn(self, train_kws):
        select_dim = train_kws['select_dim']
        # 3d numpy_array
        if select_dim == 0:
            def transpose(np_array):
                return np_array
        elif select_dim == 1:
            def transpose(np_array):
                return np_array.transpose(1, 0 ,2)
        elif select_dim == 2:
            def transpose(np_array):
                return np_array.transpose(2, 0, 1)
        else: raise Exception(f'select_dim取值有误：{select_dim}')
        return transpose

    
    def create_revert_transpose_fn(self, train_kws):
        select_dim = train_kws['select_dim']
        if select_dim == 0:
            def revert_transpose(np_array):
                return np_array
        elif select_dim == 1:
            def revert_transpose(np_array):
                return np_array.transpose(1, 0, 2)
        elif select_dim == 2:
            def revert_transpose(np_array):
                return np_array.transpose(1, 2, 0)
        else: raise Exception(f'select_dim取值有误：{select_dim}')
        return revert_transpose


    # data_tensor 是整个文件的数据
    def create_get_pixel_score_fn(self, train_kws):
        recipe = train_kws['recipe']

        if recipe == 'predict':
            see_slice = train_kws['see_slice']
            def get_pixel_score(model, data_tensor):
                rec_tensor = torch.zeros_like(data_tensor)                                                                                                 # (l, f, f)
                loss_tensor = torch.zeros_like(data_tensor)
                with torch.no_grad():
                    for i in range(data_tensor.shape[0] - see_slice):
                        input = data_tensor[i: i + see_slice, :, :].unsqueeze(0).to(DEVICE)                                              # (1, x, f, f)
                        out = model(input)                                                                                                                                          # (1, 1, f, f)
                        label = data_tensor[[i + see_slice], :, :].unsqueeze(0).to(DEVICE)
                        loss = torch.pow(out - label, 2)

                        rec_tensor[[i + see_slice]] = out[0].cpu()
                        loss_tensor[[i + see_slice]] = loss[0].cpu()
                return loss_tensor, rec_tensor

        elif recipe == 'rotate':
            from math import ceil
            batch_size = train_kws['batch_size']
            def get_pixel_score(model, data_tensor):
                rec_tensor = torch.zeros_like(data_tensor)                                                                                                 # (l, f, f)
                loss_tensor = torch.zeros_like(data_tensor)
                with torch.no_grad():
                    for i in range(ceil(data_tensor.shape[0] / batch_size)):
                        label = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1).to(DEVICE)     # (1, 1, f, f)
                        input = torch.rot90(label, 1, [2, 3])
                        rec = model(input)
                        loss = torch.pow(label - rec, 2)

                        rec_tensor[i * batch_size: (i+1) * batch_size] = rec[:, 0, :, :].cpu()                                    # (1, f, f)
                        loss_tensor[i * batch_size: (i+1) * batch_size] = loss[:, 0, :, :].cpu()
                return loss_tensor, rec_tensor

        elif recipe == 'split_rotate':
            from math import ceil
            batch_size = train_kws['batch_size']
            def get_pixel_score(model, data_tensor):
                rec_tensor = torch.zeros_like(data_tensor)                                                                                                 # (l, f, f)
                loss_tensor = torch.zeros_like(data_tensor)
                with torch.no_grad():
                    for i in range(ceil(data_tensor.shape[0] / batch_size)):
                        label = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1).to(DEVICE)
                        a, b = label.chunk(2, 2)
                        a1, a2 = a.chunk(2, 3)
                        b1, b2 = b.chunk(2, 3)
                        out = []
                        for item in list([a1, a2, b1, b2]):
                            input = torch.rot90(item, 1, [2, 3])
                            rec_split = model(input)
                            out.append(rec_split)

                        a_out = torch.cat((out[0], out[1]), 3)
                        b_out = torch.cat((out[2], out[3]), 3)
                        rec = torch.cat((a_out, b_out), 2)
                        loss = torch.pow(label - rec, 2)
                        
                        rec_tensor[i * batch_size: (i+1) * batch_size] = rec[:, 0, :, :].cpu()
                        loss_tensor[i * batch_size: (i+1) * batch_size] = loss[:, 0, :, :].cpu()
                return loss_tensor, rec_tensor

        else:
            from math import ceil
            batch_size = train_kws['batch_size']
            def get_pixel_score(model, data_tensor):
                rec_tensor = torch.zeros_like(data_tensor)                               # (l, f, f)
                loss_tensor = torch.zeros_like(data_tensor)
                with torch.no_grad():
                    for i in range(ceil(data_tensor.shape[0] / batch_size)):
                        label = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1).to(DEVICE)     # (1, 1, f, f)
                        input = label
                        rec = model(input)
                        loss = torch.pow(input - rec, 2)

                        rec_tensor[i * batch_size: (i+1) * batch_size] = rec[:, 0, :, :].cpu()
                        loss_tensor[i * batch_size: (i+1) * batch_size] = loss[:, 0, :, :].cpu()
                return loss_tensor, rec_tensor

        return get_pixel_score


    # data_tensor 是整个文件的数据
    def create_get_sample_score_fn(self, train_kws):
        recipe = train_kws['recipe']

        if recipe == 'predict':
            see_slice = train_kws['see_slice']
            def get_sample_score(model, data_tensor):
                slice_scores = []
                with torch.no_grad():
                    for i in range(data_tensor.shape[0] - see_slice):
                        input = data_tensor[i: i+see_slice].unsqueeze(0).to(DEVICE)
                        out = model(input)
                        label = data_tensor[[i+see_slice]].unsqueeze(0).to(DEVICE)
                        loss = torch.mean(torch.pow(out - label, 2), dim=(1, 2, 3))
                        slice_scores.append(loss.item())
                return np.max(slice_scores)

        elif recipe == 'rotate':
            from math import ceil
            batch_size = train_kws['batch_size']
            def get_sample_score(model, data_tensor):
                slice_scores = []
                with torch.no_grad():
                    for i in range(ceil(data_tensor.shape[0] / batch_size)):
                        label = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1).to(DEVICE)
                        input = torch.rot90(label, 1, [2, 3])
                        rec = model(input)
                        loss = torch.mean(torch.pow(label - rec, 2), dim=(1, 2, 3))
                        slice_scores += loss.cpu().tolist()
                return np.max(slice_scores)

        elif recipe == 'split_rotate':
            from math import ceil
            batch_size = train_kws['batch_size']
            def get_sample_score(model, data_tensor):
                slice_scores = []
                with torch.no_grad():
                    for i in range(ceil(data_tensor.shape[0] / batch_size)):
                        label = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1).to(DEVICE)
                        a, b = label.chunk(2, 2)
                        a1, a2 = a.chunk(2, 3)
                        b1, b2 = b.chunk(2, 3)
                        out = []
                        for item in list([a1, a2, b1, b2]):
                            input = torch.rot90(item, 1, [2, 3])
                            rec_split = model(input)
                            out.append(rec_split)

                        a_out = torch.cat((out[0], out[1]), 3)
                        b_out = torch.cat((out[2], out[3]), 3)
                        rec = torch.cat((a_out, b_out), 2)
                        loss = torch.mean(torch.pow(label - rec, 2), dim=(1, 2, 3))

                        slice_scores += loss.cpu().tolist()
                return np.max(slice_scores)
                
        else:
            from math import ceil
            batch_size = train_kws['batch_size']
            def get_sample_score(model, data_tensor):
                slice_scores = []
                with torch.no_grad():
                    for i in range(ceil(data_tensor.shape[0] / batch_size)):
                        input = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1).to(DEVICE)
                        rec = model(input)
                        loss = torch.mean(torch.pow(input - rec, 2), dim=(1, 2, 3))
                        slice_scores += loss.cpu().tolist()
                return np.max(slice_scores)

        return get_sample_score