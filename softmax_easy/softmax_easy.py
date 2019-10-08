import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs = 784
num_outputs = 10
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()#调用父类(超类)
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x):# x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))#转换数据
        return y
'''
module的call里面调用module的forward方法
forward里面如果碰到Module的子类，回到第1步，如果碰到的是Function的子类，继续往下
'''
net = LinearNet(num_inputs, num_outputs)

