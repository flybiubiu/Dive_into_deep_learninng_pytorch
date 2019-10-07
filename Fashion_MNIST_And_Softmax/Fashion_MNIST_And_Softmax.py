import torch
import torchvision
import torchvision.transforms as tranforms
import matplotlib.pylab as plt
import time
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import numpy as np
import gzip
import os

def load_mnist(path, kind = 'train'):
    '''load 手动下载的数据集'''
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype = np.uint8, offset = 8)
        #此函数将缓冲区解释为一维数组。 暴露缓冲区接口的任何对象都用作参数来返回
        '''TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

        [offset] [type]          [value]          [description] 
        0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
        0004     32 bit integer  60000            number of items 
        0008     unsigned byte   ??               label 
        0009     unsigned byte   ??               label 
        ........ 
        xxxx     unsigned byte   ??               label
        The labels values are 0 to 9.

        TRAINING SET IMAGE FILE (train-images-idx3-ubyte):

        [offset] [type]          [value]          [description] 
        0000     32 bit integer  0x00000803(2051) magic number 
        0004     32 bit integer  60000            number of images 
        0008     32 bit integer  28               number of rows 
        0012     32 bit integer  28               number of columns 
        0016     unsigned byte   ??               pixel 
        0017     unsigned byte   ??               pixel 
        ........ 
        xxxx     unsigned byte   ??               pixel'''
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype = np.uint8,offset = 16).reshape(len(labels), 784)
        return images, labels

def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
        X_train, y_train = load_mnist('./', kind = 'train')
        X_test, y_test = load_mnist('./', kind = 't10k')
        X_train_tensor = torch.from_numpy(X_train).to(torch.float32).view(-1, 1, 28, 28) * (1 / 255.0)
        X_test_tensor = torch.from_numpy(X_test).to(torch.float32).view(-1, 1, 28, 28) * (1 / 255.0)
        y_train_tensor = torch.from_numpy(y_train).to(torch.int64).view(-1, 1)
        y_test_tensor = torch.from_numpy(y_test).to(torch.int64).view(-1, 1)
        import torch.utils.data as Data
        mnist_train = Data.TensorDataset(X_train_tensor, y_train_tensor)
        mnits_test = Data.TensorDataset(X_test_tensor, y_test_tensor)

        if sys.platform.startswith('win'):
            num_workers = 0#0表示不用额外的进程来加速读取数据
        else:
            num_workers = 24
        train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = num_workers)
        test_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle = False, num_workers = num_workers)
        return train_iter, test_iter
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
#初始化模型参数
num_inputs = 784#28*28
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype = torch.float)
b = torch.zeros(num_outputs, dtype = torch.float)
W.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)
#实现softmax运算
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim = 1, keepdim = True)
    return X_exp / partition #广播机制
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
#定义交叉熵损失
def cross_entropy(y_hat, y):#y_hat是样本在类别上的预测概率
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))#0作用在行,1作用在列
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))
#计算单样本的准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
print(accuracy(y_hat, y))
#在整个数据集上评估
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim = 1) == y.squeeze(1)).float().mean().item()#squeeze转换为秩为1
        n += y.shape[0]#第行的长度
    return acc_sum / n
print ("init evaluate_accuray: ", evaluate_accuracy(test_iter, net))

def sgd(parans, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size
