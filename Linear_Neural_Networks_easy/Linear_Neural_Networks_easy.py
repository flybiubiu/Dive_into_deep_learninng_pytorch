import torch
import numpy as np
import random
import torch.nn as nn


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype = torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size = labels.size()), dtype = torch.float)

import torch.utils.data as Data
batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle = True)
for X, y in data_iter:
    print (X, y)
    break

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()#super调用父类，解决多重继承
        self.linear = nn.Linear(n_feature, 1)#n_feature, 1 指的是维度
    # forward定义向前传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print (net)# 使用print可以打印出网络结构

#写法一
net = nn.Sequential(nn.Linear(num_inputs, 1))#Sequential是一个有序的容器,网络层将按照在传入Sequential的顺序依次被添加到计算图中
#写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
#写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, 1))]))

print (net)
print (net[0])

for param in net.parameters():
     print (param)

from torch.nn import init
init.normal_(net[0].weight, mean = 0, std = 0.01)
init.constant_(net[0].bias, val = 0)#也可以直接修改bias的data:net[0].bias.data.fill_(0)
loss = nn.MSELoss()#均方损失函数 (xi-yi)^2

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr = 0.03)
print (optimizer)
'''
params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
lr (float) – 学习率
momentum (float, 可选) – 动量因子（默认：0）
weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认：0）
dampening (float, 可选) – 动量的抑制因子（默认：0）
nesterov (bool, 可选) – 使用Nesterov动量（默认：False)
'''
#为不同的子网络设置不同的学习率,这在finetune时经常用到
#optimizer = optim.SGD([{'params': net.subnet1.parameters()}, '''lr = 0.03''' ,{'params': net.subnet2.parameters(), 'lr': 0.01}], lr = 0.03)#如果对某个参数不指定学习率,就是用最外层的默认学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 1#学习速率为之前的几倍
print(optimizer)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:#data_iter上面的Data已经划分好
        output = net(X)
        l = loss(output, y.view(-1, 1))#-1表示不确定的数
        optimizer.zero_grad()#梯度清零,等价于net.zero_grad()
        l.backward()
        optimizer.step()#用来迭代模型参数,我们在step函数中指明批量大小,从而对批量中样本梯度求平均
    print ('epoch %d, loss: %f' % (epoch, l.item()))

dense = net[0]
print (true_w, dense.weight)
print (true_b, dense.bias)