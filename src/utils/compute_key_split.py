"""
This file provides the util func to compute key split.
"""
import numpy as np

def split_by_keys(data_X, data_y, train_ratio=0.8, val_ratio = 0.1, test_ratio = 0.1, seed = 31):

    assert abs (train_ratio + val_ratio + test_ratio - 1.00) < 1e-6 , "Set ratios must sum to 1.00 !"
    keys = list(data_y.keys())
    np.random.seed(seed)
    np.random.shuffle(keys)

    split_point1 = int(len(keys)*train_ratio)
    split_point2 = int(len(keys)- len(keys)*test_ratio)
    train_keys = keys[:split_point1]
    val_keys = keys[split_point1:split_point2]
    test_keys = keys[split_point2:]

    return train_keys, val_keys, test_keys


