import numpy as np
import matplotlib.pyplot as plt

# Gradient Descend Algorithm
x_data = [0, 1, 2, 3]
y_data = [0, 3, 6, 9]
w = 1.0
cost_list = []
epoch_list = []


def forward(x):
    return (w * x)


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_predict = forward(x)
        cost += (y_predict-y) ** 2
    return cost / len(xs)


def gradient(xs, ys, a):
    total = 0
    for x, y in zip(xs, ys):
        total += (2 * x * (x * w - y))
    total = (total / len(xs)) * a
    return total


print("Before Training: ", 4, forward(4))


for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data, 0.01)
    w -= grad_val
    print("Epoch: ", epoch, "w = ", w, "loss = ", cost_val, "gradient = ", grad_val)
    cost_list.append(cost_val)
    epoch_list.append(epoch)


plt.plot(epoch_list, cost_list)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()
