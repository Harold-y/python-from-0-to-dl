import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.], [2.], [3.], [4.]])
y_data = torch.Tensor([[3.], [6.], [9.], [12.]])
epoch_list = []
loss_list = []


class LinearReg(torch.nn.Module):

    def __init__(self):
        super(LinearReg, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 创建x, y都为一维的线性模型 (create a model whose x and y are all one-dimensional)

    def forward(self, x):
        y_pred = self.linear(x)  # 计算y prediction (compute y prediction)
        return y_pred


model = LinearReg()  # 创建出模型
criterion = torch.nn.MSELoss(reduction='sum')  # 用criterion计算MSE Loss，不计算平均值 (using MSE Loss function without average
# value)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 用SGD作为优化器 (Using SGD as our optimizer)

for epoch in range(100):
    y_pred = model(x_data)  # 计算y prediction (compute y predict)
    loss = criterion(y_pred, y_data)  # 计算Loss (compute Loss)
    print('Epoch: ', epoch, 'Loss: ', loss.item())

    optimizer.zero_grad()  # 清空optimizer (Clear Optimizer)
    loss.backward()  # Loss反向传播 (Loss back Propagation)
    optimizer.step()  # 优化权重 (optimize our weight)
    epoch_list.append(epoch)
    loss_list.append(loss.item())


print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
x_test = torch.Tensor([[5.0]])
y_test = model(x_test)
print('y_predict = ', y_test)

plt.plot(epoch_list, loss_list)
plt.ylabel("Loss")
plt.xlabel("Training Epoch")
plt.show()
