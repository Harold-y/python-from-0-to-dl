import numpy as np
import torch
import matplotlib.pyplot as plt

xy = np.loadtxt('../resources/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
epoch_list = []
loss_list = []


class MultipleDimModel(torch.nn.Module):
    def __init__(self):
        super(MultipleDimModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activation = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = MultipleDimModel()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('Epoch -', epoch, 'Loss: ', loss.item())
    loss_list.append(loss.item())
    epoch_list.append(epoch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
