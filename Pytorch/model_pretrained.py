import torchvision.datasets
from torch import nn

# train_data = torchvision.datasets.ImageNet("./data_imagenet", split="train",
#                                            transform=torchvision.transforms.ToTensor())
from torchvision.models import VGG16_Weights

vgg16_false = torchvision.models.vgg16()
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)    # 下载地址在环境变量 TORCH_HOME中
print("ok")
print(vgg16_true)
train_data = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor())

# 添加
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# vgg16_true.classifier[7] = nn.Linear(10,1)
# print(vgg16_true)

# 修改
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
