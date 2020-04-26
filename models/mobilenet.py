from torchvision.models import mobilenet_v2
from torch import nn

from models.basic_module import BasicModule

class MobileNetV2(BasicModule):
    def __init__(self):
        super(MobileNetV2,self).__init__()
        self.model_name = 'MobileNetV2'
        self.model = mobilenet_v2(pretrained=True)
        self.model.num_classes = 10
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.model.num_classes),
        )

    def forward(self,input):
        return self.model(input)