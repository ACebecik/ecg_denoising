"""
This file provides the util fnc to generate train/test/val sets by stacking segments.
"""

import torch
import numpy as np
from src.utils.dict_loader import DictLoader
from src.utils.compute_key_split import split_by_keys


def stack_data(data_X, data_y, keys):

    stacked_segments = []
    stacked_labels = []
    for key in keys:
        seg = torch.tensor(data_X[key], dtype=torch.float32)
        stacked_segments.append(seg)
        label = torch.tensor(data_y[key], dtype=torch.float32)
        stacked_labels.append(label)

    stacked_segments = torch.cat(stacked_segments,dim=0 )
    stacked_labels = torch.cat(stacked_labels, dim=0)
    return stacked_segments, stacked_labels

def stack_all_sets(data_X, data_y, train_key_list, val_key_list,test_key_list):
    train_X, train_y = stack_data(data_X, data_y, train_key_list)
    val_X, val_y = stack_data(data_X, data_y, val_key_list)
    test_X, test_y = stack_data(data_X, data_y, test_key_list)

    return (train_X, train_y), (val_X, val_y), (test_X, test_y)

if __name__ == "__main__":
    loader = DictLoader()
    data_X, data_y = loader.load(training_task="classification",snr=1)

    train_key_list, val_key_list, test_key_list = split_by_keys(data_X, data_y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                                                    seed=42)

    (train_X, train_y), (val_X, val_y), (test_X, test_y) = stack_all_sets(data_X, data_y, train_key_list, val_key_list,test_key_list)

