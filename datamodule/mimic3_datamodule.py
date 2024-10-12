import torch as th
import numpy as np
import pickle as pkl
from data_config import HCV_Num_Possible_Values
from typing import List
from torch.utils.data import Dataset, DataLoader
from task_config import Task_Config
from sklearn.model_selection import StratifiedKFold
import random


class MIMIC3Data():
    _hcv_num_possible_values = HCV_Num_Possible_Values.values()
    _eye_mat_list = [np.eye(num, dtype=np.float32)
                     for num in _hcv_num_possible_values]

    def __init__(self) -> None:
        self.icustay_id = [0]
        self.fpath = [0]
        self.hcv = th.empty((1, 0, 0))
        self.nume = th.empty((1, 0, 0))
        self.wave = th.empty((1, 0, 0))
        self.demography = th.empty((1, 2))
        self.mortality = th.empty((1, 1))

    @classmethod
    def onehot_encoder(cls, hcv_category):
        hcv_category_list = []
        for idx, varr in enumerate(hcv_category):
            val = cls._eye_mat_list[idx][varr].copy()
            hcv_category_list.append(val)

        hcv_category_oh = np.concatenate(hcv_category_list, 1)
        return hcv_category_oh.T

    @classmethod
    def new_obj_from_datadict(cls, icustay_id, fpath, datadict: dict,
                              use_less_wave_var=False):
        data = cls()
        data.icustay_id = [icustay_id]
        data.fpath = [fpath]
        numeric = datadict['numeric'][[1, 2, 4, 9, 10, 11], :]
        category = datadict['category']
        # numeric = datadict['numeric']
        # category = datadict['category']
        category_oh = cls.onehot_encoder(category)
        hcv = np.vstack([numeric, category_oh])
        nume_seg = datadict['nume_seg']
        wave_seg = datadict['wave_seg']
        if use_less_wave_var:
            wave_seg = wave_seg[:2, :]
            
        data.hcv = th.from_numpy(hcv).unsqueeze(0)
        data.nume = th.from_numpy(nume_seg).unsqueeze(0)
        data.wave = th.from_numpy(wave_seg).unsqueeze(0)
        data.demography = th.tensor([[datadict['AGE'], datadict['GENDER']]],
                                    dtype=th.float32)
        data.mortality = th.tensor([[datadict['MORTALITY_INUNIT']]],
                                   dtype=th.float32)
        return data

    @classmethod
    def batch_from_data_list(cls, data_list: 'List[MIMIC3Data]'):
        batch = cls()
        batch.fpath = [data.fpath[0] for data in data_list]
        batch.icustay_id = [data.icustay_id[0] for data in data_list]
        hcv_list = [data.hcv for data in data_list]
        batch.hcv = th.cat(hcv_list, 0)
        nume_list = [data.nume for data in data_list]
        batch.nume = th.cat(nume_list, 0)
        wave_list = [data.wave for data in data_list]
        batch.wave = th.cat(wave_list, 0)
        demography_list = [data.demography for data in data_list]
        batch.demography = th.cat(demography_list, 0)
        mortality_list = [data.mortality for data in data_list]
        batch.mortality = th.cat(mortality_list, 0)
        return batch

    def move_to(self, device):
        self.hcv = self.hcv.to(device)
        self.nume = self.nume.to(device)
        self.wave = self.wave.to(device)
        self.demography = self.demography.to(device)
        self.mortality = self.mortality.to(device)
        return self


class MIMIC3Dataset(Dataset):
    def __init__(self, for_training, desc_list,
                 labels, multiple_negative, root_dir,
                 sample_last_k, use_less_wave_var) -> None:
        self._for_training = for_training
        self._root_dir = root_dir
        self._sample_last_k = sample_last_k
        self._use_less_wave_var = use_less_wave_var
        if for_training:
            self._multiple_negative = multiple_negative
            self._pos_desc_list = [
                desc_list[idx]
                for idx, label in enumerate(labels)
                if label == 1]
            self._neg_desc_list = [
                desc_list[idx]
                for idx, label in enumerate(labels)
                if label == 0]
            assert len(self._neg_desc_list) > \
                len(self._pos_desc_list) * multiple_negative
            num_pos = len(self._pos_desc_list)
            num_total = num_pos + len(self._neg_desc_list)
            print(f"Training set: total {num_total}, postive {num_pos}")
        else:
            self._desc_list = []
            self._labels = []
            for desces, label in zip(desc_list, labels):
                if label == 0:
                    if self._sample_last_k > 0:
                        desces = desces[:self._sample_last_k]
                    self._desc_list.extend(desces)
                    self._labels.extend([label] * len(desces))
                else:
                    self._desc_list.append(desces[0])
                    self._labels.append(label)

            num_total = len(self._labels)
            num_pos = sum(self._labels)
            print(f"Test set: total {num_total}, postive {num_pos}")

    def __len__(self,):
        if self._for_training:
            return (self._multiple_negative + 1) * len(self._pos_desc_list)
        else:
            return len(self._desc_list)

    def __getitem__(self, index) -> MIMIC3Data:
        if self._for_training:
            if index % (self._multiple_negative + 1) == 0:
                idx = int(index // (self._multiple_negative + 1))
                desces = self._pos_desc_list[idx]
                # desc = random.choice(desces)
                desc = desces[0]
                label = 1
            else:
                idx = int(index // (self._multiple_negative + 1))
                jdx = idx * (self._multiple_negative)
                kdx = jdx + index % (self._multiple_negative + 1)
                desces = self._neg_desc_list[kdx]
                if self._sample_last_k > 0:
                    desces = desces[:self._sample_last_k]
                    # may should to be refactored
                desc = random.choice(desces)
                label = 0
        else:
            desc = self._desc_list[index]
            label = self._labels[index]

        subject_id = desc['SUBJECT_ID']
        icustay_id = desc['ICUSTAY_ID']
        base_los2 = desc['seg_base_los2']
        end_los2 = desc['seg_end_los2']
        fpath = f"{self._root_dir}/{label}_{subject_id}_" + \
            f"{icustay_id}_{base_los2:.0f}_{end_los2:.0f}.pkl"
        with open(fpath, 'rb') as rbfile:
            datadict = pkl.load(rbfile)

        datadict['MORTALITY_INUNIT'] = label
        mimic3data = MIMIC3Data.new_obj_from_datadict(
            icustay_id, fpath, datadict, 
            self._use_less_wave_var)
        return mimic3data


class MIMIC3DataModule():
    def __init__(self, task_config: Task_Config, fold_idx, multiple_negative,
                 batch_size, num_workers, n_splits, sample_last_k,
                 use_less_wave_var=False) -> None:
        self._data_dir = task_config.data_dir
        print('data root dir', self._data_dir)
        desc_path = f'{self._data_dir}/icustay__desc_dict_list.pkl'
        with open(desc_path, 'rb') as rbfile:
            icustay__desc_dict_list = pkl.load(rbfile)

        label_path = f'{self._data_dir}/icustay__label_list.pkl'
        with open(label_path, 'rb') as rbfile:
            icustay__label_list = pkl.load(rbfile)

        self.num_samples = len(icustay__desc_dict_list)
        self.num_mortality_samples = sum(icustay__label_list)
        skf = StratifiedKFold(n_splits=n_splits)
        splits = list(skf.split(icustay__desc_dict_list, icustay__label_list))
        train_index, test_index = splits[fold_idx]
        self._train_desc_list = [icustay__desc_dict_list[idx]
                                 for idx in train_index]
        self._train_labels = [icustay__label_list[idx] for idx in train_index]
        assert multiple_negative < (
            len(self._train_labels) / sum(self._train_labels))
        self._test_desc_list = [icustay__desc_dict_list[idx]
                                for idx in test_index]
        self._test_labels = [icustay__label_list[idx] for idx in test_index]
        self.num_total_test_samples = len(self._test_labels)
        self.num_test_mortality_samples = sum(self._test_labels)
        self._multiple_negative = multiple_negative
        self._sample_last_k = sample_last_k
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train_loader = None
        self._test_loader = None
        self._use_less_wave_var = use_less_wave_var

    def train_loader(self,):
        indexs = np.random.permutation(len(self._train_labels))
        desc_list = [self._train_desc_list[idx] for idx in indexs]
        labels = [self._train_labels[idx] for idx in indexs]
        dataset = MIMIC3Dataset(True, desc_list, labels,
                                self._multiple_negative,
                                self._data_dir, self._sample_last_k,
                                self._use_less_wave_var)
        loader = DataLoader(dataset, self._batch_size,
                            num_workers=self._num_workers,
                            collate_fn=MIMIC3Data.batch_from_data_list)
        return loader

    def test_loader(self,):
        if self._test_loader is None:
            dataset = MIMIC3Dataset(False, self._test_desc_list,
                                    self._test_labels,
                                    self._multiple_negative,
                                    self._data_dir,
                                    self._sample_last_k,
                                    self._use_less_wave_var)
            loader = DataLoader(dataset, self._batch_size,
                                num_workers=self._num_workers,
                                collate_fn=MIMIC3Data.batch_from_data_list)
            self._test_loader = loader

        return self._test_loader


if __name__ == "__main__":
    device = th.device('cuda:3')

    config_name = 'test'
    datamodule = MIMIC3DataModule(config_name, 0, 3, 64, 0, 4, 3)
    # datamodule.train_loader()

    for batch in datamodule.train_loader():
        batch.move_to(device)
        print(th.sum(batch.mortality).item(), end=' ')

    # batch = next(datamodule.train_loader)
    print()
