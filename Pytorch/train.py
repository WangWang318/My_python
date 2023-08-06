import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch import nn
from mymodel import MyNet

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f'训练数据集的长度{train_data_size}')
print(f'测试数据集的长度{test_data_size}')

print(train_data[0][0].shape)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络模型
net = MyNet()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

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
    for data in train_dataloader:
        images, targets = data
        outputs = net(images)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        print(f'训练次数: {total_train_step}, Loss: {loss.item()}')

    with torch.no_grad():
        