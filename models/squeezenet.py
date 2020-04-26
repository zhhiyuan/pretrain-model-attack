from torchvision.models import squeezenet1_0, squeezenet1_1
from torch import nn

from models.basic_module import BasicModule

class SqueezeNet1_0(BasicModule):
    def __init__(self):
        super(SqueezeNet1_0, self).__init__()

        self.model_name = 'SqueezeNet1_0'
        self.model = squeezenet1_0(pretrained=True)
        self.model.num_classes = 10
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, self.model.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self,input):
        return self.model(input)


class SqueezeNet1_1(BasicModule):
    def __init__(self):
        super(SqueezeNet1_1,self)

        self.model_name = 'SqueezeNet1_1'
        self.model = squeezenet1_1(pretrained=True)
        self.model.num_classes = 10
        self.model.classifier =  nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, self.model.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self,input):
        return self.model(input)