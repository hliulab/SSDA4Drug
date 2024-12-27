import gzip
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader


def get_labeled_dataloaders(x_source, y_source, x_target, y_target, batch_size,i):
    x_source = x_source.T
    x_target = x_target.T
    y_target = y_target["response"]
    train_source_dataloader, test_source_dataloader = get_source_dataloader_generator(x_source,y_source,batch_size,i)
    ## target data process
    x_target = torch.FloatTensor(x_target.values)
    y_target = torch.LongTensor(y_target.values)
    test_target_dateset = TensorDataset(x_target,y_target)
    test_target_dataloader = DataLoader(test_target_dateset,
                                        batch_size=batch_size,
                                        shuffle=True)

    return train_source_dataloader, test_source_dataloader, test_target_dataloader


def get_source_dataloader_generator(x_source, y_source,batch_size,i):
    ## 5折交叉验证
    y_source = y_source["response"]
    x_source = x_source.T
    x_train_source, x_test_source, y_train_source, y_test_source = train_test_split(x_source, y_source,
                                                                                  test_size=0.2, random_state=i)
    x_train_source = torch.FloatTensor(x_train_source.values)
    x_test_source = torch.FloatTensor(x_test_source.values)
    y_train_source = torch.LongTensor(y_train_source.values)
    y_test_source = torch.LongTensor(y_test_source.values)
    train_source_dateset = TensorDataset(x_train_source,y_train_source)
    test_source_dateset = TensorDataset(x_test_source,y_test_source)

    train_source_dataloader = DataLoader(train_source_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    test_source_dataloader = DataLoader(test_source_dateset,
                                        batch_size=batch_size,
                                        shuffle=True)
    return train_source_dataloader, test_source_dataloader
