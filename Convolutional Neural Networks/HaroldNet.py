import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as F
from matplotlib import pyplot as plt

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

train_data = datasets.CIFAR10(root='../resources/MNIST', train=True, transform=transform, download=False)
test_dataset = datasets.CIFAR10(root='../resources/MNIST', train=False, transform=transform, download=False)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

accuracy_list = []
epoch_list = []
loss_list = []


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1,
                               padding=1)  # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
        self.conv2 = nn.Conv2d(16, 32, 3, 1,
                               padding=1)  # We double the feature maps for every conv layer as in pratice it is really good.
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(4 * 4 * 64,
                             500)  # I/p image size is 32*32, after 3 MaxPooling layers it reduces to 4*4 and 64 because our last conv layer has 64 outputs. Output nodes is 500
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)  # output nodes are 10 because our dataset have 10 different categories

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply relu to each output of conv layer.
        x = F.max_pool2d(x, 2, 2)  # Max pooling layer with kernal of 2 and stride of 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)  # flatten our images to 1D to input it to the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Applying dropout b/t layers which exchange highest parameters. This is a good practice
        x = self.fc2(x)
        return x


model = LeNet()
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
    for epoch in range(15):
        train(epoch)
        epoch_list.append(epoch)
        run_test_sets()
    torch.save(model, '../models/CIFAR10/CIFAR10_20210116_2.pk1')
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
