import numpy as np
import matplotlib.pyplot as plt

# Gradient Descend Algorithm
x_data = [0, 1, 2, 3]
y_data = [0, 3, 6, 9]
w = 1.0
cost_list = []
epoch_list = []


def forward(x):
    return x * w


def loss(x, y):
    return (forward(x) - y) ** 2


def gradient(x, y):
    return 2 * x * (x * w - y)


print("Before Training: ", 4, forward(4))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        gra_val = gradient(x, y)
        w -= 0.01 * gra_val
        print("X = ", x, "Y = ", y, "Gradient = ", gra_val)
    print("Epoch: ", epoch, "w = ", w, "loss = ", loss_val)
    cost_list.append(loss_val)
    epoch_list.append(epoch)


plt.plot(epoch_list, cost_list)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()


