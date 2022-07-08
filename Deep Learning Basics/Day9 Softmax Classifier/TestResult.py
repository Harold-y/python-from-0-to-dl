import torch
from mnist import *
import glob
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import torchvision
from skimage import io, transform
from SoftmaxClassifier import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('../models/MNIST/MNIST_test1.pk1')  # 加载模型
model = model.to(device)
model.eval()  # 把模型转为test模式
# 以灰度图的方式读取要预测的图片
img = cv2.imread('img/mnist10.png', 0)
img = cv2.resize(img, (28, 28))
# 因为MNIST数据集中的图片都是黑底白字，所以此处还需要图片进行反色处理。
height, width = img.shape
dst = np.zeros((height, width), np.uint8)
for i in range(height):
    for j in range(width):
        dst[i, j] = 255 - img[i, j]

img = dst
# 处理完成后的图片和之前的步骤就一样了，送入网络，输出结果
img=np.array(img).astype(np.float32)
img=np.expand_dims(img,0)
img=np.expand_dims(img,0)#扩展后，为[1，1，28，28]
img=torch.from_numpy(img)
img = img.to(device)
output=model(Variable(img))
prob = F.softmax(output, dim=1)
prob = Variable(prob)
prob = prob.cpu().numpy()  #用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
print(prob)  #prob是10个分类的概率
pred = np.argmax(prob) #选出概率最大的一个
print(pred.item())