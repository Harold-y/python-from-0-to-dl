import torch

x_list = [0.0, 1.0, 2.0, 3.0]
y_list = [5.0, 10.0, 19.0, 32.0]
w1 = torch.Tensor([1.0])
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True


def forward(x):
    return (w1 * x * x) + (w2 * x) + b


def loss(x, y):
    return (forward(x) - y) ** 2


print("Before Training: ", 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(x_list, y_list):
        l = loss(x, y)
        l.backward()
        print('\tw1 grad: ', w1.grad.item(), 'w2 grad: ', w2.grad.item(), 'b grad: ', b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('Epoch: ', epoch, 'Loss: ', l.item())
    print('w1 = ', w1.data.item(), 'w2 = ', w2.data.item(), 'b = ', b.data.item())

print("After Training: ", 4, forward(4).item())
