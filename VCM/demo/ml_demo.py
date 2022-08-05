import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import numpy as np

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

# 定义一个网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.pool1 = nn.AvgPool1d(2)
        self.bn1 = nn.BatchNorm1d(3)
        self.fc1 = nn.Linear(12, 4)

    
    def forward(self, x):
        x = self.pool1(x)
        x = self.bn1(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)

        return x
    
# 定义网络
model = net()

# 定义loss
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# 定义训练数据
x = torch.randn((3, 3, 8))

# model.fc1.weight.requires_grad = False  # fc1.weight不计算梯度
# model.fc1.bias.requires_grad = False  # fc1.weight不计算梯度

print(model.fc1.weight.grad)
print(model.fc1.bias.grad)  # fc1.bias计算梯度

output = model(x)
target = torch.tensor([1, 1, 1])
loss = loss_fn(output, target)

loss.backward()

print(model.fc1.weight.grad)
print(model.fc1.bias.grad)

print(model.bn1.weight.grad)
print(model.bn1.bias.grad)