import torch as t
class Config:
    model_path = 'ShuffleNetV2_ep1.pth'# 预训练模型，None表示重新训练
    model = 'ShuffleNetV2'#加载的模型，模型名必须与models/__init__.py中的名字一致
    '''
    ShuffleNetV2,MobileNetV2,SqueezeNet1_0,SqueezeNet1_1
    VGG11,VGG13,VGG16,VGG19
    ResNet18,ResNet34,ResNet50
    '''
    lr = 0.0005 #学习率
    use_gpu = True  #是否使用gpu
    MEAN=(0.485, 0.456, 0.406)
    STD=(0.229, 0.224, 0.225)#均值和方差
    train_epoch = 1  # 将数据集训练多少次
    save_every = 1  # 每训练多少轮保存一次模型
    # imagenet得出的较好的值，具体过程参考
    # https://cloud.tencent.com/developer/ask/153881

    test_num = 2000  # 选择攻击和测试的样本数量


    batch_size = 128  # 每次喂入多少数据

    print_freq = 500  # 每训练多少批次就打印一次

    def _parese(self):
        self.device = t.device('cuda') if self.use_gpu else t.device('cpu')
        print('Caculate on {}'.format(self.device))
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))
