from time import strftime

import torch as t
import os
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
from torch.optim import Adam
import tqdm

import models
from attack.PGDAttack import LinfPGDAttack
from attack.PixelAttack import attack_all
from config import Config
DOWNLOAD_ImageNet = False #是否下载数据集



def train():
    '''
    训练神经网络
    :return:
    '''
    #1.加载配置
    opt = Config()
    opt._parese()
    global DOWNLOAD_ImageNet

    #1a.加载模型
    model = getattr(models,opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device)

    #2.定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=opt.lr)


    #3.加载数据
    if not (os.path.exists('./data/ImageNet/')) or not os.listdir('./data/ImageNet/'):
        DOWNLOAD_ImageNet=True

    transform = tv.transforms.Compose([
        #要先完成所有操作，然后totensor，normalize
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=opt.MEAN, std=opt.STD)
    ])
    train_data = tv.datasets.ImageNet(
        root='./data/ImageNet/',
        split='val',
        transform=transform,
        download=DOWNLOAD_ImageNet
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers
                              )

    #训练模型
    for epoch in range(opt.train_epoch):
        for ii,(data,label) in tqdm.tqdm(enumerate(train_loader)):
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()

            if (ii+1)%opt.print_freq ==0:
                print('loss:%.2f'%loss.cpu().data.numpy())
        if (epoch + 1) % opt.save_every == 0:
            model.save(epoch=epoch+1)

@t.no_grad()
def test_acc():
    #测试准确率
    # 1. 加载配置
    opt = Config()
    opt._parese()
    global DOWNLOAD_ImageNet
    if not (os.path.exists('./data/ImageNet/')) or not os.listdir('./data/ImageNet/'):
        DOWNLOAD_ImageNet=True

    # 1a.加载模型
    model = getattr(models, opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device)
    model.eval()

    # 2.加载数据
    transform = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=opt.MEAN, std=opt.STD)

    ])

    test_data = tv.datasets.ImageNet(
        root='./data/ImageNet/',
        split='val',
        transform=transform,
        download=DOWNLOAD_ImageNet
    )
    #test_data.loader
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=opt.num_workers)

    correct_num = 0
    for ii, (test_x, test_y) in enumerate(test_loader):
        if ii >= opt.test_num:
            break
        test_x = test_x.to(opt.device)
        test_score = model(test_x)
        correct = (t.argmax(test_score.to('cpu'), 1) == test_y).numpy()
        correct_num = correct_num + correct[0]

    accuracy = correct_num/opt.test_num
    print('test accuracy:%.2f' % accuracy)
    return accuracy

@t.no_grad()
def attack_model_pixel():
    '''
    pixel攻击模型
    :return:
    '''

    # 1.加载配置
    opt = Config()
    opt._parese()
    global DOWNLOAD_ImageNet
    if not (os.path.exists('./data/ImageNet/')) or not os.listdir('./data/ImageNet/'):
        DOWNLOAD_ImageNet=True

    # 1a.加载模型
    model = getattr(models, opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device).eval()


    # 2.加载数据
    transform = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=opt.MEAN,std= opt.STD)
    ])

    test_data = tv.datasets.ImageNet(
        root='./data/ImageNet/',
        split='val',
        transform=transform,
        download=DOWNLOAD_ImageNet
    )

    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)



    success_rate = attack_all(model,test_loader,pixels=3,targeted=False,maxiter=100,popsize=400,verbose=False,device=opt.device,sample=opt.test_num)
    accuracy = test_acc()
    accuracy_after = accuracy*(1-success_rate)
    string = 'pixel , {} , {} , {} , {} ,  {}\n'.format(opt.model_path,accuracy,accuracy_after,success_rate, strftime('%m_%d_%H_%M_%S'))
    open('log.csv','a').write(string)


def attack_model_PGD():
    '''
    PGD攻击模型
    :return:
    '''

    # 1.加载配置
    opt = Config()
    opt._parese()
    global DOWNLOAD_ImageNet
    if not (os.path.exists('./data/ImageNet/')) or not os.listdir('./data/ImageNet/'):
        DOWNLOAD_ImageNet=True

    # 1a.加载模型
    model = getattr(models, opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device).eval()


    # 2.加载数据
    transform = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=opt.MEAN,std= opt.STD)
    ])

    test_data = tv.datasets.ImageNet(
        root='./data/ImageNet/',
        split='val',
        transform=transform,
        download=DOWNLOAD_ImageNet
    )

    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    success_num = 0
    attack = LinfPGDAttack(model)
    for ii, (data, label) in enumerate(test_loader):
        if ii>=opt.test_num:
            break

        test_score = model(data.to(opt.device))
        if t.argmax(test_score.to('cpu'), 1) == label.to('cpu'):
            continue
        perturb_x =attack.perturb(data.numpy(),label)
        test_score = model(t.FloatTensor(perturb_x).to(opt.device))
        if t.argmax(test_score.to('cpu'), 1) != label:
            success_num+=1


    success_rate = success_num/ii
    accuracy = test_acc()
    accuracy_after = accuracy*(1-success_rate)
    string = 'PGD , {} , {} , {} , {} ,  {}\n'.format(opt.model_path, accuracy, accuracy_after, success_rate,strftime('%m_%d_%H_%M_%S'))
    open('log.csv', 'a').write(string)

if __name__ == '__main__':
    test_acc()
    # attack_model_pixel()
    # attack_model_PGD()
    #  train()
    #  test_acc()
