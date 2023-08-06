import torch
import torchvision.models

# # 方式1 加载模型
# model1 = torch.load("vgg16_method1.pth")
# print(model1)

# 方式2 加载模型
vgg16 = torchvision.models.vgg16()
model = torch.load("vgg16_method2.pth")
vgg16.load_state_dict(model)
print(vgg16)
