import torch
import matplotlib
import numpy as np


x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [3.0, 5.0, 9.0, 12.0]
w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x):
    return x * w


def loss(x, y):
    return (forward(x) - y) ** 2


print('before training: ', 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
    print('Epoch:', epoch, 'loss: ', l.item())


print('after training: ', 4, forward(4).item())

