# %% ../../nbs/common.modules.ipynb 3
import math
import os
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import itertools
import random

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import warnings

import time

def MSE_loss(prediction_samples, labels) :
    N = labels.shape[0]
    prediction_samples, _ = torch.sort(prediction_samples, axis = 1)
    labels, _ = torch.sort(labels, axis = 1)
    return torch.sum(torch.square(prediction_samples - labels)) / N

def QLIKE_loss(prediction_samples, labels) :
    N = labels.shape[0]
    prediction_samples = prediction_samples.view(-1).contiguous()
    labels = labels.view(-1).contiguous()
    return torch.sum(labels/ prediction_samples - torch.log(labels/ prediction_samples) - 1) / N

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss




def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


