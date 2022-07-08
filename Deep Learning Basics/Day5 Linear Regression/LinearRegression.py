import torch
import matplotlib.pyplot as plt
import numpy as np

x_data = torch.Tensor([[1.], [2.], [3.], [4.]])
y_data = torch.Tensor([[3.], [6.], [9.], [12.]])
epoch_list = []
loss_list = []

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(50):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_list.append(epoch)
    loss_list.append(loss.item())

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[5.]])
y_test = model(x_test)
print('y_pred = ', y_test.data)


plt.plot(epoch_list, loss_list)
plt.ylabel("Loss")
plt.xlabel("Training Epoch")
plt.show()
