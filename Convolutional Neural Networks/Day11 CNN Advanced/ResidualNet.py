import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as f

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='../resources/MNIST', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../resources/MNIST', train=False, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = f.relu(self.conv1(x))
        y = self.conv2(y)
        return f.relu(x + y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.reblock1 = ResidualBlock(16)
        self.reblock2 = ResidualBlock(32)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(f.relu(self.conv1(x)))
        x = self.reblock1(x)
        x = self.mp(f.relu(self.conv2(x)))
        x = self.reblock2(x)
        x = x.view(in_size, -1)
        return self.fc(x)


model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.5, lr=0.01)


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


if __name__ == '__main__':
    for epoch in range(7):
        train(epoch)
        run_test_sets()
    torch.save(model, '../models/MNIST/MNIST_test_20210529_1.pk1')
