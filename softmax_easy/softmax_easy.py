import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs = 784
num_outputs = 10
#class LinearNet(nn.Module):
#    def __init__(self, num_inputs, num_outputs):
#        super(LinearNet, self).__init__()#调用父类(超类)
#        self.linear = nn.Linear(num_inputs, num_outputs)
#    def forward(self, x):# x shape: (batch, 1, 28, 28)
#        y = self.linear(x.view(x.shape[0], -1))#转换数据
#        return y
'''
module的call里面调用module的forward方法
forward里面如果碰到Module的子类，回到第1步，如果碰到的是Function的子类，继续往下
'''
#net = LinearNet(num_inputs, num_outputs)
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
from collections import OrderedDict
net = nn.Sequential(
        # FlattenLayer(),
        # nn.Linear(num_inputs, num_outputs)
        OrderedDict([
          ('flatten', FlattenLayer()),
          ('linear', nn.Linear(num_inputs, num_outputs))])
        )

#from collections import OrderedDict
#collections为python提供了一些加强版的数据结构
#OrderedDict 可以理解为有序的dict，底层源码是通过双向链表来实现，每一个元素为一个map存储key-value
#net = nn.Sequential(OrderedDict([('flatten', FlattenLayer()), ('linear', nn.Linear(num_inputs, num_outputs))]))
#一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
#init.normal_(net.linear.weight, mean = 0, std = 0.01)#正态分布初始化
#init.constant_(net.linear.bias, val = 0)#常数初始化
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
#softmax和交叉熵损失函数
loss = nn.CrossEntropyLoss()#分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定,这里使用Pytorch内置的稳定性更好
#定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
#训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, nudef evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim = 1) == y.squeeze(1)).float().mean().item()#squeeze转换为秩为1
        n += y.shape[0]#第行的长度
    return acc_sum / nm_epochs, batch_size, None, None, optimizer)