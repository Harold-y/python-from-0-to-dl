import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x_data = [0, 1, 2, 3]
y_data = [2.7, 5.7, 8.7, 11.7]


def forward(x):
    return (x * w) + b


def loss_function(x, y):
    y_predict = forward(x)
    return (y_predict - y) * (y_predict - y)


w_list = np.arange(0.0, 4.0, 0.1)
b_list = np.arange(0.0, 3.0, 0.1)
[w, b] = np.meshgrid(w_list, b_list)

loss_val = 0
for x_val, y_val in zip(x_data, y_data):
    y_predict_value = forward(x_val)
    loss_value = loss_function(x_val, y_val)
    loss_val += loss_value

# print(np.meshgrid(w_list, b_list))
print(w)
print(b)
print(loss_val)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('w value')
ax.set_ylabel('b value')
ax.set_zlabel('Loss function value')
ax.plot_surface(w, b, loss_val)

plt.show()
