import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys



num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))#loc=0,均值,float=1,标准差
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size = labels.size()))
print (features[0], labels[0])

def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize = (3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)#1代表标量
plt.savefig("./Linear_Neural_Networks/one_dimensional_label.jpg")#打印第二个特征features[:, 1]和标签labels的线性关系
plt.scatter(features[:, 0].numpy(), labels.numpy(), 1)#1代表标量
plt.savefig("./Linear_Neural_Networks/two_dimensional_label.jpg")#打印第二个特征features[:, 1]和标签labels的线性关系

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) #样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):#start:0,stop:num_examples,步长:batch_size
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])#最后一次可能不足一个batch,LongTensor(64-bit integer (signed))
        yield features.index_select(0, j), labels.index_select(0,j)#0表示维度,yield类似return,并保留函数当前的运行状态，等待下一次的调用

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print (X, y)
    break

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)))
dtype = torch.float64
b = torch.zeros(1, dtype = torch.float64)
w.requires_grad_(requires_grad = True)#需要对这些参数求梯度来迭代参数的值
b.requires_grad_(requires_grad = True)

def linreg(X, w, b):
    return torch.mm(X, w) + b#torch.mul(a, b)是矩阵a和b对应位相乘,a和b的维度必须相等,torch.mm(a, b)是矩阵a和b矩阵相乘

def squared_loss(y_hat, y):#注意这里返回的是向量,另外,pytorch里的MSELoss并没有除以2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):#小批量梯度下降
    for param in params:
        param.data -= lr * param.grad / batch_size #注意这里更改param时用的param.data

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):#训练模型一共需要num_epochs个迭代周期
    #在每个迭代周期中,会使用训练数据集中所有样本一次,X,y分别是小批量样本的特征和标签
    for X,y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()#l是有关小批量X和y的损失
        l.backward()#小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)#使用小批量随机梯度下降迭代模型参数

        #梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print ('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print (true_w, '\n', w)
print (true_b, '\n', b)