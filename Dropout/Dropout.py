import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    #这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return  torch.zeros_like(X)
    mask = (torch.randn(X.shape) < keep_prob).float()
    return mask * X / keep_prob

X = torch.arange(16).view(2, 8)
print (dropout(X, 0))
print (dropout(X, 0.5))
print (dropout(X, 1.0))

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = torch.tensor(np.random.normal(0, 0.01, size = (num_inputs, num_hiddens1)), dtype = torch.float, requires_grad = True)
b1 = torch.zeros(num_hiddens1, requires_grad = True)
W2 = torch.tensor(np.random.normal(0, 0.01, size = (num_hiddens1, num_hiddens2)), dtype = torch.float, requires_grad = True)
b2 = torch.zeros(num_hiddens2, requires_grad = True)
W3 = torch.tensor(np.random.normal(0, 0.01, size = (num_hiddens2, num_outputs)), dtype = torch.float, requires_grad = True)
b3 = torch.zeros(num_outputs, requires_grad = True)
params = [W1, b1, W2, b2, W3, b3]

drop_prob1, drop_prob2 = 0.2, 0.5
def net(X, is_training = True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:#只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1)# 在第一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2)#在第二层全连接后添加丢弃层
    return torch.matmul(H2, W3) + b3

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
#其第一个参数为对象，第二个为类型名或类型名的一个列表。其返回值为布尔型。若对象的类型与参数二的类型相同则返回True。若参数二为一个元组，则若对象类型与元组中类型名之一相同即返回True。
            net.eval()#评估模式,这会关闭dropout
            acc_sum += (net(X).argmax(dim  = 1) == y).float().sum().item()
            net.train()#改回训练模式
        else:#自定义模型
            if('is_training' in net.__code__.co_varnames):#如果有is_training这个参数
                #将is_training设置为False
                acc_sum += (net(X, is_training = False).argmax(dim = 1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim = 1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)