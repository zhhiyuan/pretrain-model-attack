from torchvision.models import resnet18,resnet34,resnet50
from torch import nn

from models.basic_module import BasicModule

class ResNet18(BasicModule):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.model_name = 'ResNet18'

        self.model = resnet18(pretrained=True)


    def forward(self,input):
        return self.model(input)


class ResNet34(BasicModule):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model_name = 'ResNet34'

        self.model = resnet34(pretrained=True)

    def forward(self, input):
        return self.model(input)

class ResNet50(BasicModule):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model_name = 'ResNet50'

        self.model = resnet50(pretrained=True)


    def forward(self, input):
        return self.model(input)