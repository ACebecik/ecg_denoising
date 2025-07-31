"""
This file creates the TensorDataset and Dataloader onj for the stacked tensors of train/test/val sets.
"""

from torch.utils.data import TensorDataset, DataLoader

def create_dataset(train, val, test):

    (train_X, train_y) = train
    (val_X, val_y) = val
    (test_X, test_y) = test

    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    test_ds = TensorDataset(test_X, test_y)

    return train_ds, val_ds, test_ds

def create_dataloaders(train_ds, val_ds, test_ds, batch_size = 64):

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_dl, val_dl, test_dl

