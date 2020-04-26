from torchvision.models import squeezenet1_0, squeezenet1_1
from torch import nn

from models.basic_module import BasicModule

class SqueezeNet1_0(BasicModule):
    def __init__(self):
        super(SqueezeNet1_0, self).__init__()

        self.model_name = 'SqueezeNet1_0'
        self.model = squeezenet1_0(pretrained=True)

    def forward(self,input):
        return self.model(input)


class SqueezeNet1_1(BasicModule):
    def __init__(self):
        super(SqueezeNet1_1,self)

        self.model_name = 'SqueezeNet1_1'
        self.model = squeezenet1_1(pretrained=True)


    def forward(self,input):
        return self.model(input)