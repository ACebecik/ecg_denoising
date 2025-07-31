"""This file implements the CustomDataset class used to feed network the data via DataLoader."""
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, x, y):
        # data loading
        self.x = x
        self.y = y

    def __getitem__(self, index):
        # allows for indexing
        get_x = self.x[index]
        get_y = self.y[index]
        return get_x,get_y

    def __len__(self):
        return len(self.x)