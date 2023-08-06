"""
    网络模型
    数据(输入、标注)
    损失函数
    cuda()
"""

import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter


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


train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f'训练数据集的长度{train_data_size}')
print(f'测试数据集的长度{test_data_size}')

print(train_data[0][0].shape)

train_dataloader = DataLoader(train_data, batch_size=256)
test_dataloader = DataLoader(test_data, batch_size=256)

# 添加tensorboard
writer = SummaryWriter("./logs_train") # tensorboard --logdir=Pytorch//logs_train

# 创建网络模型
net = MyNet()
if torch.cuda.is_available():
    net = net.cuda()
    print("cuda")

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 0

for i in range(10):
    print(f'-------------第{i + 1}轮训练-------------')

    # 训练开始
    net.train()
    for data in train_dataloader:
        images, targets = data
        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()

        outputs = net(images)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 50 == 0:
            print(f'训练次数: {total_train_step}, Loss: {loss.item()}')
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试部分开始
    net.eval()
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            images, targets = data
            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()

            outputs = net(images)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的accuracy:{total_accuracy / test_data_size}")

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(net, f"net_{i}.pth")
    print("模型已保存")

writer.close()

