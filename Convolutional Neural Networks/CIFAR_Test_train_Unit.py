import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as F
from matplotlib import pyplot as plt
from DenseNet1 import DenseBlock
from DenseNet1 import TransitionLayer
from DenseNet1 import DenseNet

batch_size = 64
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_data = datasets.CIFAR10(root='../resources/MNIST', train=True, transform=transform_train, download=False)
test_dataset = datasets.CIFAR10(root='../resources/MNIST', train=False, transform=transform, download=False)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

accuracy_list = []
epoch_list = []
loss_list = []

model = DenseNet(nr_classes=10)
# model = HaroldNet(in_channels=3, square_resolution=32, n_class=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.8, lr=0.01)

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

        if batch_index % 200 == 199:
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
    accuracy_list.append(100 * correct / total)


if __name__ == '__main__':
    for epoch in range(12):
        train(epoch)
        epoch_list.append(epoch)
        run_test_sets()
    torch.save(model, '../models/CIFAR10/CIFAR10_20210118_1.pk1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on Test Sets')
    plt.grid()
    plt.plot(epoch_list, accuracy_list)
    plt.show()

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.plot(epoch_list, loss_list)
    plt.show()

