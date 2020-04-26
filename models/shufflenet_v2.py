from torchvision.models import shufflenet_v2_x1_0

from models.basic_module import BasicModule

class ShuffleNetV2(BasicModule):
    def __init__(self):
        super(ShuffleNetV2,self).__init__()

        self.model_name = 'ShuffleNetV2'
        self.model = shufflenet_v2_x1_0(pretrained=True)
        self.model.num_classes = 10

    def forward(self,input):
        return self.model(input)
