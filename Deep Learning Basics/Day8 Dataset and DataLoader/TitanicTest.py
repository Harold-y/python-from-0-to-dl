import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd


class TitanicDataset(Dataset):
    def __init__(self, filepath):
        # xy = np.loadtxt(filepath, delimiter=',', skiprows=1, dtype=bytes).astype(str)
        xy = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13), dtype=str)
        # print(xy)
        self.x_data = xy[1, :-1]
        self.y_data = xy[:, [0]]
        print(self.x_data)
        print('-'*30)
        print(self.y_data)
        self.len = xy.shape[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


dataset = TitanicDataset('../resources/titanic/train.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=12)


class TitanicModel(torch.nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.linear1 = torch.nn.Linear(10, 8)
        self.linear2 = torch.nn.Linear(8, 6)
        self.linear3 = torch.nn.Linear(6, 4)
        self.linear4 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


model = TitanicModel()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_list = []
epoch_list = []


def plot():
    plt.plot(epoch_list, loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    for epoch in range(20):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print('Epoch: ', epoch, 'Loss: ', loss.item())
            epoch_list.append(epoch)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    plot()
