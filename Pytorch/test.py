import torch
import torchvision.transforms
from PIL import Image
from torch import nn

image_path = "img/dog.png"
image = Image.open(image_path)
print(image)

image = image.convert("RGB")
print(image)

transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()
])

image = transform(image)
image = image.cuda()
print(image.shape)


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = torch.load("net_9.pth")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))

# 测试
model.eval()
with torch.no_grad(): # 节约内存， 提升性能
    output = model(image)
print(output)
print(output.argmax(1))
