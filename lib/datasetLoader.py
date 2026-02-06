from __future__ import division
import numpy as np
import torch
import os
import logging
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler

logger = logging.getLogger('Dataset load')

class DatasetLoad(Dataset):
    def __init__(self, data_path, flag):
        
        self.x = np.load(os.path.join(data_path, f'{flag}_data.npy'))
        self.label = np.load(os.path.join(data_path, f'{flag}_label.npy'))
        self.len = self.label.shape[0]

        logger.info(f'dataset len: {self.len}')
        logger.info(f'building datasets from {data_path}...')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index],  self.label[index]


class Dataset_all(object):
    def __init__(self, configs):
        self.configs = configs
        self.train_set = DatasetLoad(data_path=configs.data_path, flag = 'train')
        self.val_set = DatasetLoad(data_path=configs.data_path, flag = 'validation')
        self.test_set = DatasetLoad(data_path=configs.data_path, flag = 'test')
        # print(len(self.test_set))
        self.train_loader = DataLoader(self.train_set, batch_size=configs.batch_size, sampler=RandomSampler(self.train_set), num_workers = configs.num_workers)
        self.validation_loader = DataLoader(self.val_set, batch_size=1024, shuffle=False, num_workers = configs.num_workers)
        self.test_loader = DataLoader(self.test_set, batch_size=1024, shuffle=False, num_workers = configs.num_workers)

    def _get_data(self, flag = 'train'):
        if flag == 'train' :
            return self.train_set, self.train_loader
        elif flag == 'val' :
            return self.val_set, self.validation_loader
        elif flag == 'test' :
            return self.test_set, self.test_loader
        else :
            logger.info(f'- Error Dataset Flag -')