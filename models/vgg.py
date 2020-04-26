from torchvision.models import vgg11,vgg13,vgg16,vgg19
from torch import nn

from models.basic_module import BasicModule


class VGG11(BasicModule):#继承的类可以保存和读取模型
    def __init__(self):
        super(VGG11,self).__init__()
        self.model_name = 'VGG11'
        #获取预训练模型的结构和权重
        self.model = vgg11(pretrained=True)
        #改模型分类器为十分类
        self.model.num_classes = 10 #此处需要跳转到源码查看模型是否具有num_classes分类个数的参数


    #前向传播
    def forward(self,input):
        return self.model(input)

class VGG13(BasicModule):
    def __init__(self):
        super(VGG13,self).__init__()
        self.model_name = 'VGG13'
        #获取预训练模型的结构和权重
        self.model = vgg13(pretrained=True)
        #改模型分类器为十分类
        self.model.num_classes = 10#此处需要跳转到源码查看模型


    #前向传播
    def forward(self,input):
        return self.model(input)


class VGG16(BasicModule):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model_name = 'VGG16'
        # 获取预训练模型的结构和权重
        self.model = vgg16(pretrained=True)
        # 改模型分类器为十分类
        self.model.num_classes = 10  # 此处需要跳转到源码查看模型


    # 前向传播
    def forward(self, input):
        return self.model(input)

class VGG19(BasicModule):
    def __init__(self):
        super(VGG19, self).__init__()
        self.model_name = 'VGG19'
        # 获取预训练模型的结构和权重
        self.model = vgg19(pretrained=True)
        # 改模型分类器为十分类
        self.model.num_classes = 10  # 此处需要跳转到源码查看模型


    # 前向传播
    def forward(self, input):
        return self.model(input)