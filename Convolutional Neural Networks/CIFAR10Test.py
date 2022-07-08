import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as F
from matplotlib import pyplot as plt
from HaroldNet import LeNet
import matplotlib.image as mpimg  # mpimg 用于读取图片
import PIL.ImageOps
import requests
from PIL import Image
import numpy as np
from skimage.transform import resize

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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


def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()  # This process will happen in normal cpu.
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('../models/CIFAR10/CIFAR10_20210118_1.pk1')  # 加载模型
model = model.to(device)
model.eval()


def evaluation(srcPath):
    src = mpimg.imread(srcPath)  # 读取和代码处于同一目录下的 图片
    # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理

    # resize至32x32
    newImg = resize(src, output_shape=(32, 32))
    plt.imshow(newImg)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    plt.show()

    # 转换为（B,C,H,W）大小
    imagebatch = newImg.reshape(-1, 3, 32, 32)

    # 转换为torch tensor
    image_tensor = torch.from_numpy(imagebatch).float()

    # 调用模型进行评估
    model.eval()
    output = model(image_tensor.to(device))
    _, predicted = torch.max(output.data, 1)
    pre = predicted.cpu().numpy()
    print(pre)  # 查看预测结果ID
    print(classes[pre[0]])


def setEvaluation():
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images = images.to(device)
    labels = labels.to(device)
    output = model(images)
    _, preds = torch.max(output, 1)

    fig = plt.figure(figsize=(25, 4))

    for idx in np.arange(0, 20, 1):
        # ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
        ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
        plt.imshow(im_convert(images[idx]))
        ax.set_title("{} ({})".format(str(classes[preds[idx].item()]), str(classes[labels[idx].item()])),
                     color=("green" if preds[idx] == labels[idx] else "red"))
    plt.plot()
    plt.show()


if __name__ == '__main__':
    setEvaluation()

    for seq in range(3, 4):
        srcPath = 'img/CIFAR_' + str(seq) + '.jpg'
        evaluation(srcPath)
        time.sleep(5)

