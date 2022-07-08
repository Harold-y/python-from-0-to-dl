import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from matplotlib import pyplot as plt
import numpy as np

loss_list = []
epoch_list = []
accuracy_list = []
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch_pooling = nn.Conv2d(in_channels, 24, kernel_size=1)

        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pooling(branch_pool)

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, dim=1)


class ResidualBlockFour(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockFour, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        return F.relu(x + y)


class ResidualBlockTwo(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockTwo, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class HaroldNet(nn.Module):
    def __init__(self, in_channels, square_resolution, n_class):
        super(HaroldNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(88, 44, kernel_size=3)
        self.conv4 = nn.Conv2d(44, 20, kernel_size=1)
        self.mp = nn.MaxPool2d(2)
        self.inception1 = InceptionA(16)
        self.inception2 = InceptionA(32)
        self.residual1 = ResidualBlockTwo(44)
        self.residual2 = ResidualBlockFour(20)
        self.fc = nn.Linear(20, n_class)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.inception1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.inception2(x)
        x = self.mp(F.relu(self.conv3(x)))
        x = self.residual1(x)
        x = self.mp(F.relu(self.conv4(x)))
        x = self.residual2(x)
        x = x.view(in_size, -1)
        return self.fc(x)


model = HaroldNet(in_channels=3, square_resolution=32, n_class=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.5, lr=0.01)
batch_size = 64
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_data = datasets.CIFAR10(root='../resources/MNIST', train=True, transform=transform, download=False)
test_dataset = datasets.CIFAR10(root='../resources/MNIST', train=False, transform=transform, download=False)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        # forward, backward, update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_index % 300 == 299:
            print('[%d, %d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 300))
            running_loss = 0.0
    loss_list.append(running_loss)


def run_test_sets():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicts = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()

    print('Accuracy on test sets: %d%%' % (100 * correct / total))
    print('Total/Correct: [', total, '/', correct, ']')
    accuracy_list.append(correct / total)


if __name__ == '__main__':
    for epoch in range(40):
        train(epoch)
        run_test_sets()
        epoch_list.append(epoch)
        if accuracy_list[epoch] >= 0.73:
            torch.save(model, '../models/CIFAR10/CIFAR10_20210112_'+str(epoch)+'.pk1')

    plt.plot(epoch_list, loss_list)
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(epoch_list, accuracy_list)
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
