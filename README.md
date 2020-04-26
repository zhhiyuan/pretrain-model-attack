# pretrain-model-attack
 adversarial attack using torchvision pretrained model on ciafr10
使用torchvision的预训练模型进行生成对抗攻击，数据集为cifar10

### 攻击方法(attack mode)

- PGD attack
code from [there](https://github.com/wanglouis49/pytorch-adversarial_box)
epsilon=0.3, k=40, a=0.01

- pixel attack

pixel = 3, maxiter=100,targeted=False, popsize=400, verbose=False

code from [there](https://github.com/DebangLi/one-pixel-attack-pytorch)

### 目前模型(model)

- VGG11,VGG13,VGG16,VGG19

- SqueezeNet1.0,SqueezeNet1.1

- MobileNetV2

- ResNet18,ResNet34,ResNet50

- ShuffleNetV2

模型来自`torchvision.model`，更多细节[参考](https://pytorch.org/docs/stable/torchvision/models.html)

### 环境

- torch==1.1.0

- torchvision==0.3.0

- pillow<7.0.0

- tqdm

### 运行须知

- 运行`main.py`即可，此方法需要配置`config.py`文件。或使用notebook运行`main.ipynb`，此方法不需要配置文件，在ipynb里面配置即可

- 若`./ckps`文件夹下无预训练模型，则需要先训练模型

- 先训练模型(`main.py`中的train()函数，若有预训练模型，可以跳过)，模型训练完并保存在ckps文件夹后，修改config中model_path为预训练路径

- 实施攻击(`main.py`中attack_model_PGD()或attack_model_pixel()函数)，之后会保存所有内容在`log.txt`文件夹下，保存的内容有
       
       - 攻击类型
       - 模型名
       - 准确率(攻击前)
       - 准确率(攻击后)
       - 攻击成功率
       - 保存日志时间
