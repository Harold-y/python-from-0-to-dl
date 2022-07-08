import numpy as np
import torch
import matplotlib.pyplot as plt

xy = np.loadtxt('../resources/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
epoch_list = []
loss_list = []


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # self.active = torch.nn.ReLU()  # 激活函数 Activation Function, ReLu -> [0, 1]
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))  # To ensure the result is in (0, 1)
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('Epoch: ', epoch, 'Loss: ', loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()  # Clear Optimizer
    loss.backward()  # Back Propagation
    optimizer.step()  # Update


plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
