from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torch

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)


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


net = MyNet()
loss = nn.CrossEntropyLoss()

step = 0
for data in dataloader:
    images, targets = data
    output = net(images)
    result_loss = loss(output, targets)
    result_loss.backward()
    # print(result_loss)