import abc
import os
from math import ceil
import time
from tqdm import tqdm

from tqdm import tqdm
import torch
from trixi.util.pytorchexperimentstub import PytorchExperimentStub

from example_algos.data.numpy_dataset import get_numpy2d_dataset, DataPreFetcher


class Algorithm:
    def __init__(self, basic_kws, train_kws):
        self.__dict__.update(basic_kws)
        self.__dict__.update(train_kws)

        log_dict = {}
        if self.logger is not None: log_dict = {0: self.logger}
        self.tx = PytorchExperimentStub(name=self.model_type, base_dir=self.log_dir, config=None, loggers=log_dict,)


    def train(self):
        print('train')

        n_items = None
        train_loader = get_numpy2d_dataset(
            base_dir=self.train_data_dir,
            num_processes=self.batch_size,
            pin_memory=True,
            batch_size=self.batch_size,
            mode="train",
            target_size=self.target_size,
            drop_last=True,
            n_items=n_items,
            functions_dict=self.dataset_functions,
        )
        val_loader = get_numpy2d_dataset(
            base_dir=self.train_data_dir,
            num_processes=self.batch_size // 2,
            pin_memory=True,
            batch_size=self.batch_size,
            mode="val",
            target_size=self.target_size,
            drop_last=True,
            n_items=n_items,
            functions_dict=self.dataset_functions,
        )
        train_loader = DataPreFetcher(train_loader)
        val_loader = DataPreFetcher(val_loader)

        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss = 0

            data_loader_ = tqdm(enumerate(train_loader))
            for batch_idx, data in data_loader_:
                # data = data.cuda()
                loss = self.train_model(data)

                train_loss += loss.item()
                if batch_idx % self.print_every_iter == 0:
                    status_str = (
                        f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} "
                        f" ({100.0 * batch_idx / len(train_loader):.0f}%)] Loss: "
                        f"{loss.item():.6f}"
                    )
                    data_loader_.set_description_str(status_str)

                    cnt = epoch * len(train_loader) + batch_idx
                    self.tx.add_result(loss.item(), name="Train-Loss", tag="Losses", counter=cnt)

                    # if self.logger is not None:
                        # self.tx.l[0].show_image_grid(inpt, name="Input", image_args={"normalize": True})
                        # self.tx.l[0].show_image_grid(inpt_rec, name="Reconstruction", image_args={"normalize": True})

            print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader):.6f}")

            # validate
            self.model.eval()

            val_loss = 0
            data_loader_ = tqdm(enumerate(val_loader))
            data_loader_.set_description_str("Validating")
            for _, data in data_loader_:
                loss = self.eval_model(data)
                val_loss += loss.item()

            self.tx.add_result(
                val_loss / len(val_loader), name="Val-Loss", tag="Losses", counter=(epoch + 1) * len(train_loader))
            print(f"====> Epoch: {epoch} Validation loss: {val_loss / len(val_loader):.6f}")

        self.tx.save_model(self.model, "model")
        time.sleep(2)


    def train_model(self, data):
        input, label = self.get_input_label(data)
        loss = self.calculate_loss(self.model, input, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


    def eval_model(self, data):
        with torch.no_grad():
            input, label = self.get_input_label(data)
            loss = self.calculate_loss(self.model, input, label)
        return loss


    def predict(self, **kwargs):
        print('predict')
        from .nifti_io import ni_save, ni_load
        from .function import save_images, init_validation_dir

        test_dir = self.test_dir
        _, pred_pixel_dir, pred_sample_dir = init_validation_dir(algo_name=self.name, dataset_dir=test_dir)
        test_dir = os.path.join(test_dir, 'data')

        test_dir_list = os.listdir(test_dir)
        length = len(test_dir_list)
        handle = tqdm(enumerate(test_dir_list))

        has_num = 'num' in kwargs.keys()
        if has_num: num = kwargs['num']
        return_rec = kwargs['return_rec'] if 'return_rec' in kwargs.keys() else False
        
        return_rec = True

        self.model.eval()
        for i, f_name in handle:
            if not f_name.startswith('n2'): continue

            if has_num:
                if i == num: break
            ni_file_path = os.path.join(test_dir, f_name)
            ni_data, ni_aff = ni_load(ni_file_path)

            # pixel
            result = self.score_pixel_2d(ni_data, return_rec=return_rec)
            save_images(pred_pixel_dir, f_name, ni_aff, score=result['score'], ori=result['ori'], rec=result['rec'])

            # sample
            sample_score = self.score_sample_2d(ni_data)
            with open(os.path.join(pred_sample_dir, f_name + ".txt"), "w") as target_file:
                target_file.write(str(sample_score))

            handle.set_description_str(f'predict: {i+1}/{length}')


    def validate(self):
        print('validate')
        from .function import init_validation_dir
        from scripts.evalresults import eval_dir

        test_dir = self.test_dir
        score_dir, pred_pixel_dir, pred_sample_dir = init_validation_dir(algo_name=self.name, dataset_dir=test_dir)

        # pixel
        pred_pixel_dir = os.path.join(pred_pixel_dir, 'score')
        eval_dir(pred_dir=pred_pixel_dir, label_dir=os.path.join(test_dir, 'label', 'pixel'), mode='pixel', save_file=os.path.join(score_dir, 'pixel'))

        # sample
        eval_dir(pred_dir=pred_sample_dir, label_dir=os.path.join(test_dir, 'label', 'sample'), mode='sample', save_file=os.path.join(score_dir, 'sample'))


    def statistics(self):
        print('statistics')
        from .nifti_io import ni_load, ni_save
        import matplotlib.pyplot as plt
        import numpy as np

        test_dir = self.test_dir
        predict_dir = os.path.join(test_dir, 'eval', self.name, 'predict')
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


    def score_pixel_2d(self, np_array, return_score=True, return_ori=False, return_rec=False):
        score = None;   ori = None; rec = None
        np_array = self.transpose(np_array)

        ori_shape = np_array.shape
        # to_transforms = torch.nn.Upsample((self.target_size, self.target_size), mode="bilinear")
        from_transforms = torch.nn.Upsample((ori_shape[1], ori_shape[2]), mode="bilinear")

        data_tensor = torch.from_numpy(np_array).float()
        label_tensor = torch.nn.Upsample((self.target_size, self.target_size), mode="bilinear")(data_tensor[None])[0].cuda() # resolution
        data_tensor = self.to_transforms(data_tensor[None])[0].cuda()

        score_tensor, rec_tensor = self.get_pixel_score(self.model, data_tensor, label_tensor=label_tensor)

        if return_score:
            score_tensor = from_transforms(score_tensor[None])[0]
            score = score_tensor.detach().cpu().numpy()
            score = self.revert_transpose(score)
        if return_ori:
            data_tensor = from_transforms(data_tensor[None])[0]
            ori = data_tensor.detach().numpy()
            ori = self.revert_transpose(ori)
        if return_rec:
            rec_tensor = from_transforms(rec_tensor[None])[0]
            rec = rec_tensor.detach().cpu().numpy()
            rec = self.revert_transpose(rec)

        return {'score': score, 'ori': ori, 'rec': rec}


    def score_sample_2d(self, np_array):
        # to_transforms = torch.nn.Upsample((self.target_size, self.target_size), mode="bilinear")
        data_tensor = torch.from_numpy(np_array).float()
        label_tensor = torch.nn.Upsample((self.target_size, self.target_size), mode="bilinear")(data_tensor[None])[0].cuda() # resolution
        data_tensor = self.to_transforms(data_tensor[None])[0].cuda()

        sample_score = self.get_sample_score(self.model, data_tensor, label_tensor=label_tensor)
        return sample_score