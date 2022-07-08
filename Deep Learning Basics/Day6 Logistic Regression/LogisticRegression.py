import torch
import numpy as np
import matplotlib.pyplot as plt


x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y_data = torch.Tensor([[0], [0], [0], [1], [1], [1]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('Epoch = ', epoch, 'Loss = ', loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x_test = torch.Tensor([[4.25]])
print("Test: ", 'Hours of learning: ', x_test.data.item(), 'Possibility of passing: ', model(x_test).item())

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='purple')
plt.ylabel('Possibility of Passing')
plt.xlabel('Hours of Learning')
plt.grid()
plt.show()
