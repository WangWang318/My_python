from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torch import optim


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
opt = optim.SGD(net.parameters(), lr=0.01)

step = 0
for epoch in range(20):
    running_loss=0.0
    for data in dataloader:
        images, targets = data
        output = net(images)
        result_loss = loss(output, targets)
        # 梯度清零
        opt.zero_grad()
        # 反向传播
        result_loss.backward()
        opt.step()
        running_loss = running_loss + result_loss
    print(running_loss)