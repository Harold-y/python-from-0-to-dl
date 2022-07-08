import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd


class TitanicDataset(Dataset):
    def __init__(self, filepath):
        # xy = np.loadtxt(filepath, delimiter=',', skiprows=1, dtype=bytes).astype(str)
        xy = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(1, 2, 6, 7, 8, 10),  dtype=np.str)
        # print(xy)
        self.x_data = torch.from_numpy(xy[:, 1:-1])
        self.y_data = torch.from_numpy(xy[:, [0]])
        print(self.x_data)
        print('-'*30)
        # print(self.y_data)
        self.len = xy.shape[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


dataset = TitanicDataset('../resources/titanic/train.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=12)
