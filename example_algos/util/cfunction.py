import os

import torch
import numpy as np


class FuncFactory:

    def __init__(self):
        self.singleton = True

    def getFunction(self, fn_name, *args):
        if self.singleton and hasattr(self, fn_name):
            return getattr(self, fn_name)
        else:
            create_fn_name = 'create_' + fn_name + '_fn'
            if not hasattr(self, create_fn_name):
                raise Exception(f'工厂无法创造函数：{fn_name}')
            func = getattr(self, create_fn_name)(*args)
            setattr(self, fn_name, func)
            return func

    # 加一些训练参数，args是额外要求的训练细节
    def create_modify_train_kws_fn(self, kws, recipe):
        if recipe == 'predict':
            def modify_train_kws_fn(train_kws):
                train_kws['see_slice'] = kws['see_slice']
        elif recipe == 'mask':
            def modify_train_kws_fn(train_kws):
                train_kws['data_augment_prob'] = kws['data_augment_prob']
                train_kws['mask_square_size'] = kws['mask_square_size']
        elif recipe == 'res':
            def modify_train_kws_fn(train_kws):
                train_kws['res_size'] = kws['res_size']
                train_kws['minus_low'] = kws['minus_low']
        elif recipe == 'canny':
            def modify_train_kws_fn(train_kws):
                train_kws['canny_th'] = kws['canny_th']
                train_kws['canny_binary'] = kws['canny_binary']
        else:
            def modify_train_kws_fn(train_kws): pass

        return modify_train_kws_fn

    # 根据训练参数的情况，修改模型参数
    def create_modify_model_kws_fn(self, train_kws):
        model_type = train_kws['model_type']
        recipe = train_kws['recipe']

        func_list = []

        if recipe == 'predict':
            def modify_model_kws_fn(model_kws):
                model_kws['in_channels'] = train_kws['see_slice']
        else:
            def modify_model_kws_fn(model_kws):
                model_kws['in_channels'] = 1
        func_list.append(modify_model_kws_fn)
        
        if model_type == 'unet':
            def modify_model_kws_fn(model_kws):
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
            func_list.append(modify_model_kws_fn)
        
        def modify_model_kws(model_kws):
            for func in func_list:
                func(model_kws)
        
        return modify_model_kws


    def create_run_fn(self, run_mode):
        if run_mode == 'train':
            def run(algo, **kwargs):  algo.train(**kwargs)
        elif run_mode == 'predict':
            def run(algo, **kwargs):  algo.predict(**kwargs)
        elif run_mode == 'validate':
            def run(algo, **kwargs):  algo.validate(**kwargs)
        elif run_mode == 'statistics':
            def run(algo, **kwargs):  algo.statistics(**kwargs)
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


    # def create_transforms_fn(self, train_kws):
    #     from monai.transforms import Resize, Compose, ToTensor
    #     func_list = []
    #     common_resize = Resize((train_kws['target_size'], train_kws['target_size']))

    #     if 'resolution' in train_kws.keys():
    #         func_list.append(
    #             Resize((train_kws['resolution'], train_kws['resolution']))
    #         )
    #     func_list.append(common_resize)
    #     func_list.append(ToTensor())
    #     return Compose(func_list)


    # def create_to_transforms_fn(self, train_kws):
    #     func_list = []

    #     if 'resolution' in train_kws.keys():
    #         func_list.append(
    #             torch.nn.Upsample((train_kws['resolution'], train_kws['resolution']), mode="bilinear")
    #         )
            
    #     func_list.append(
    #         torch.nn.Upsample((train_kws['target_size'], train_kws['target_size']), mode="bilinear")
    #     )
        
    #     def to_transforms(data):
    #         for func in func_list:
    #             data = func(data)
    #         return data

    #     return to_transforms


    def create_calculate_loss_fn(self, train_kws):
        loss_type = train_kws['loss_type']

        func_list = []

        if '7loss' in loss_type:
            def  calculate_loss_fn(model, input, label):
                d0, d1, d2, d3, d4, d5, d6  = model(input)
                loss0 = torch.mean(torch.pow(d0 - label, 2))
                loss1 = torch.mean(torch.pow(d1 - label, 2))
                loss2 = torch.mean(torch.pow(d2 - label, 2))
                loss3 = torch.mean(torch.pow(d3 - label, 2))
                loss4 = torch.mean(torch.pow(d4 - label, 2))
                loss5 = torch.mean(torch.pow(d5 - label, 2))
                loss6 = torch.mean(torch.pow(d6 - label, 2))
                loss = torch.mean(loss0, loss1, loss2, loss3, loss4, loss5, loss6)
                return loss
            func_list.append(calculate_loss_fn)

        if 'l2' in loss_type:
            def calculate_loss_fn(model, input, label):
                out = model(input)
                loss = torch.mean(torch.pow(out - label, 2))
                return loss, out
            func_list.append(calculate_loss_fn)

        if 'l1' in loss_type:
            def calculate_loss_fn(model, input, label):
                out = model(input)
                loss = torch.mean(torch.abs(out - label))
                return loss, out
            func_list.append(calculate_loss_fn)

        if 'vgg' in loss_type:
            from example_algos.model.vgg import VGGLoss
            vggloss = VGGLoss()
            def calculate_loss_fn(model, input, label):
                out = model(input)
                loss = torch.mean(vggloss(
                    torch.repeat_interleave(out, repeats=3, dim=1), torch.repeat_interleave(label, repeats=3, dim=1)
                ))
                return loss, out
            func_list.append(calculate_loss_fn)
            
        # else: raise Exception(f'未知loss_type: {loss_type}')

        def calculate_loss(model, input, label):
            loss = 0
            out = 0
            for func in func_list:
                temp = func(model, input, label)
                loss += temp[0]
                out += temp[1]
            return loss, out

        return calculate_loss


    # data 是一个batch的数据，已经经过了resize处理。(16,1,f,f) gpu
    def create_get_input_label_fn(self, train_kws):
        recipe = train_kws['recipe']

        if recipe == 'predict':
            def get_input_label(data):
                input = data[:, range(train_kws['see_slice']), :, :]
                label = data[:, [train_kws['see_slice']], :, :]
                return input, label

        elif recipe == 'mask':
            import random
            from .ce_noise import get_square_mask
            def get_input_label(data):
                label = data
                if random.random() < train_kws['data_augment_prob']:
                    ce_tensor = get_square_mask(data.shape, square_size=train_kws['mask_square_size'], 
                        n_squares=1, noise_val=(torch.min(data).item(), torch.max(data).item()), data=data)
                    ce_tensor = torch.from_numpy(ce_tensor).float().cuda()
                    input_noisy = torch.where(ce_tensor != 0, ce_tensor, data)
                    input = input_noisy
                else:
                    input = label
                return input, label

        elif recipe == 'rot':
            def get_input_label(data):
                label = data
                input = torch.rot90(label, 1, [2, 3])
                return input, label

        elif recipe == 'split_rotate':
            def get_input_label(data):
                a, b = data.chunk(2, 2)
                a1, a2 = a.chunk(2, 3)
                b1, b2 = b.chunk(2, 3)
                label = torch.cat((a1, a2, b1, b2), 0)
                input = torch.rot90(label, 1, [2, 3])
                return input, label

        elif recipe == 'res': # 与低分辨率相减
            to_transform = torch.nn.Upsample((train_kws['res_size'], train_kws['res_size']), mode="bilinear")
            from_transform = torch.nn.Upsample((train_kws['origin_size'], train_kws['origin_size']), mode="bilinear")
            if train_kws['minus_low']:
                def get_input_label(data):
                    label = from_transform(to_transform(data))
                    input = label
                    return input, label
            else: # 与高分辨率相减
                def get_input_label(data):
                    label = data
                    input = from_transform(to_transform(data))
                    return input, label

        elif recipe == 'canny':
            from .function import cv2_canny
            low = train_kws['canny_th'][0]
            high = train_kws['canny_th'][1]

            if train_kws['canny_binary']:
                def get_input(input, data):
                    return input
            else:
                def get_input(input, data):
                    return torch.where(input == 0, input, data)

            def get_input_label(data):
                label = data
                data = data.cpu().numpy()
                data_slices = []
                for data_slice in data:
                    data_slice = np.squeeze(data_slice)
                    data_slice = np.expand_dims(cv2_canny(data_slice, low, high), axis=0)
                    data_slices.append(data_slice)
                input = torch.from_numpy(np.array(data_slices)).cuda()
                input = get_input(input, label)
                return input, label

        else:
            def get_input_label(data):
                input = data
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


    # data_tensor 是整个文件的数据 (d,d,d)
    def create_get_pixel_score_fn(self, train_kws):
        # recipe = train_kws['recipe']

        # if recipe == 'predict':
        #     see_slice = train_kws['see_slice']
        #     def get_pixel_score(model, data_tensor):
        #         rec_tensor = torch.zeros_like(data_tensor)                                                                                                 # (l, f, f)
        #         loss_tensor = torch.zeros_like(data_tensor)
        #         with torch.no_grad():
        #             for i in range(data_tensor.shape[0] - see_slice):
        #                 input = data_tensor[i: i + see_slice, :, :].unsqueeze(0)                                              # (1, x, f, f)
        #                 out = model(input)                                                                                                                                          # (1, 1, f, f)
        #                 label = data_tensor[[i + see_slice], :, :].unsqueeze(0)
        #                 loss = torch.pow(out - label, 2)

        #                 rec_tensor[[i + see_slice]] = out[0]
        #                 loss_tensor[[i + see_slice]] = loss[0]
        #         return loss_tensor, rec_tensor

        # elif recipe == 'rotate':
        #     from math import ceil
        #     batch_size = train_kws['batch_size']
        #     def get_pixel_score(model, data_tensor):
        #         rec_tensor = torch.zeros_like(data_tensor)                                                                                                 # (l, f, f)
        #         loss_tensor = torch.zeros_like(data_tensor)
        #         with torch.no_grad():
        #             for i in range(ceil(data_tensor.shape[0] / batch_size)):
        #                 label = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)                                   # (1, 1, f, f)
        #                 input = torch.rot90(label, 1, [2, 3])
        #                 rec = model(input)
        #                 loss = torch.pow(label - rec, 2)

        #                 rec_tensor[i * batch_size: (i+1) * batch_size] = rec[:, 0, :, :]                                                     # (1, f, f)
        #                 loss_tensor[i * batch_size: (i+1) * batch_size] = loss[:, 0, :, :]
        #         return loss_tensor, rec_tensor

        # elif recipe == 'split_rotate':
        #     from math import ceil
        #     batch_size = train_kws['batch_size']
        #     def get_pixel_score(model, data_tensor):
        #         rec_tensor = torch.zeros_like(data_tensor)                                                                                                 # (l, f, f)
        #         loss_tensor = torch.zeros_like(data_tensor)
        #         with torch.no_grad():
        #             for i in range(ceil(data_tensor.shape[0] / batch_size)):
        #                 label = label_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)
        #                 data = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)
        #                 a, b = data.chunk(2, 2)
        #                 a1, a2 = a.chunk(2, 3)
        #                 b1, b2 = b.chunk(2, 3)
        #                 out = []
        #                 for item in list([a1, a2, b1, b2]):
        #                     input = torch.rot90(item, 1, [2, 3])
        #                     rec_split = model(input)
        #                     out.append(rec_split)

        #                 a_out = torch.cat((out[0], out[1]), 3)
        #                 b_out = torch.cat((out[2], out[3]), 3)
        #                 rec = torch.cat((a_out, b_out), 2)
        #                 loss = torch.pow(label - rec, 2)
                        
        #                 rec_tensor[i * batch_size: (i+1) * batch_size] = rec[:, 0, :, :]
        #                 loss_tensor[i * batch_size: (i+1) * batch_size] = loss[:, 0, :, :]
        #         return loss_tensor, rec_tensor

        # else:
        #     from math import ceil
        #     batch_size = train_kws['batch_size']
        #     def get_pixel_score(model, data_tensor):
        #         rec_tensor = torch.zeros_like(data_tensor)                               # (l, f, f)
        #         loss_tensor = torch.zeros_like(data_tensor)
        #         with torch.no_grad():
        #             for i in range(ceil(data_tensor.shape[0] / batch_size)):
        #                 label = label_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)                     # (1, 1, f, f)
        #                 input = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)
        #                 rec = model(input)
        #                 loss = torch.pow(input - rec, 2)

        #                 rec_tensor[i * batch_size: (i+1) * batch_size] = rec[:, 0, :, :]
        #                 loss_tensor[i * batch_size: (i+1) * batch_size] = loss[:, 0, :, :]
        #         return loss_tensor, rec_tensor

        from math import ceil
        batch_size = train_kws['batch_size']
        get_input_label = self.getFunction('get_input_label', train_kws)

        def get_pixel_score(model, data_tensor, return_score=True, return_rec=False, return_input=False, return_sample_score=True):
            if return_score: score_tensor = torch.zeros_like(data_tensor)
            if return_rec: rec_tensor = torch.zeros_like(data_tensor)
            if return_input: input_tensor = torch.zeros_like(data_tensor)
            if return_sample_score: slice_scores = []

            with torch.no_grad():
                for i in range(ceil(data_tensor.shape[0] / batch_size)):
                    data = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)
                    input, label = get_input_label(data)
                    out = model(input)
                    loss = torch.pow(out - label, 2)
                    
                    if return_score: score_tensor[i * batch_size: (i+1) * batch_size] = loss[:, 0, :, :]
                    if return_rec: rec_tensor[i * batch_size: (i+1) * batch_size] = out[:, 0, :, :]
                    if return_input: input_tensor[i * batch_size: (i+1) * batch_size] = input[:, 0, :, :]
                    if return_sample_score:
                        loss = torch.mean(loss, dim=(1, 2, 3))
                        slice_scores += loss.cpu().tolist()

            result = {}
            if return_score: result['score'] = score_tensor
            if return_rec: result['rec'] = rec_tensor
            if return_input: result['input'] = input_tensor
            if return_sample_score: result['sp'] = np.max(slice_scores)

            return result

        return get_pixel_score


    # data_tensor 是整个文件的数据
    def create_get_sample_score_fn(self, train_kws):
        # recipe = train_kws['recipe']

        # if recipe == 'predict':
        #     see_slice = train_kws['see_slice']
        #     def get_sample_score(model, data_tensor):
        #         slice_scores = []
        #         with torch.no_grad():
        #             for i in range(data_tensor.shape[0] - see_slice):
        #                 label = label_tensor[[i+see_slice]].unsqueeze(0)
        #                 input = data_tensor[i: i+see_slice].unsqueeze(0)
        #                 out = model(input)
        #                 loss = torch.mean(torch.pow(out - label, 2), dim=(1, 2, 3))
        #                 slice_scores.append(loss.item())
        #         return np.max(slice_scores)

        # elif recipe == 'rotate':
        #     from math import ceil
        #     batch_size = train_kws['batch_size']
        #     def get_sample_score(model, data_tensor):
        #         slice_scores = []
        #         with torch.no_grad():
        #             for i in range(ceil(data_tensor.shape[0] / batch_size)):
        #                 label = label_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)
        #                 data = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)
        #                 input = torch.rot90(data, 1, [2, 3])
        #                 rec = model(input)
        #                 loss = torch.mean(torch.pow(label - rec, 2), dim=(1, 2, 3))
        #                 slice_scores += loss.cpu().tolist()
        #         return np.max(slice_scores)

        # elif recipe == 'split_rotate':
        #     from math import ceil
        #     batch_size = train_kws['batch_size']
        #     def get_sample_score(model, data_tensor):
        #         slice_scores = []
        #         with torch.no_grad():
        #             for i in range(ceil(data_tensor.shape[0] / batch_size)):
        #                 label = label_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)
        #                 data = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)
        #                 a, b = data.chunk(2, 2)
        #                 a1, a2 = a.chunk(2, 3)
        #                 b1, b2 = b.chunk(2, 3)
        #                 out = []
        #                 for item in list([a1, a2, b1, b2]):
        #                     input = torch.rot90(item, 1, [2, 3])
        #                     rec_split = model(input)
        #                     out.append(rec_split)

        #                 a_out = torch.cat((out[0], out[1]), 3)
        #                 b_out = torch.cat((out[2], out[3]), 3)
        #                 rec = torch.cat((a_out, b_out), 2)
        #                 loss = torch.mean(torch.pow(label - rec, 2), dim=(1, 2, 3))

        #                 slice_scores += loss.cpu().tolist()
        #         return np.max(slice_scores)
                
        # else:
        #     from math import ceil
        #     batch_size = train_kws['batch_size']
        #     def get_sample_score(model, data_tensor):
        #         slice_scores = []
        #         with torch.no_grad():
        #             for i in range(ceil(data_tensor.shape[0] / batch_size)):
        #                 label = label_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)
        #                 input = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)
        #                 rec = model(input)
        #                 loss = torch.mean(torch.pow(label - rec, 2), dim=(1, 2, 3))
        #                 slice_scores += loss.cpu().tolist()
        #         return np.max(slice_scores)

        from math import ceil
        batch_size = train_kws['batch_size']
        get_input_label = self.getFunction('get_input_label', train_kws)

        def get_sample_score(model, data_tensor):
            slice_scores = []
            with torch.no_grad():
                for i in range(ceil(data_tensor.shape[0] / batch_size)):
                    data = data_tensor[i * batch_size: (i+1) * batch_size].unsqueeze(1)
                    input, label = get_input_label(data)
                    out = model(input)
                    loss = torch.mean(torch.pow(label - out, 2), dim=(1, 2, 3))
                    slice_scores += loss.cpu().tolist()
            return np.max(slice_scores)

        return get_sample_score