#用以测试模型是否能够跑通
from models import VGG11,VGG13,VGG16,VGG19,ShuffleNetV2,ResNet18,ResNet34,ResNet50,MobileNetV2

import os
from torch import nn
from torch.optim import Adam
import torchvision as tv
from torch.utils.data import DataLoader
DOWNLOAD_CIFAR10 = False

#1.读取模型
model = MobileNetV2()

# 2.定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)


if not (os.path.exists('../data/cifar/')) or not os.listdir('../data/cifar/'):
    DOWNLOAD_CIFAR10 = True

transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    #cifar10得出的较好的值，具体过程参考
    #https://cloud.tencent.com/developer/ask/153881
    tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_data = tv.datasets.CIFAR10(root='../data/cifar/',
        train=True,
        transform=transforms,
        download=DOWNLOAD_CIFAR10)
dl = DataLoader( dataset=train_data,
        batch_size=4,
        shuffle=True,
        num_workers=0)
for ii, (data, label) in enumerate(dl):


    optimizer.zero_grad()
    score = model(data)
    loss = criterion(score, label)
    loss.backward()
    optimizer.step()

    break
