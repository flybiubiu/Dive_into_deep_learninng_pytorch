import torch
import torchvision
import torchvision.transforms as tranforms
import matplotlib.pylab as plt
import time
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import numpy as np

def load_data():
    '''load 手动下载的数据集'''
    base = "./"
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    paths = []
    for fname in files:
        paths.append(get_file(fname, origin = base + fname))
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset = 8)
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
    with gzip.open(paths[1], 'rb') as imgpath:
        X_train = np.frombuffer(imgpath.read(), np.uint8, offset = 16).reshape(len(y_train), 28, 28)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset = 8)
    with gzip.open(paths[3], 'rb') as imgpath:
        X_test = np.frombuffer(imgpath.read(), np.uint8, offset = 16).reshape(len(y_test), 28, 28)
    import torch.utils.data as Data
    mnist_train = Data.TensorDataset(X_train, y_train)
    mnits_test = Data.TensorDataset(X_test, y_test)

    if sys.platform.startswith('win'):
        num_workers = 0#0表示不用额外的进程来加速读取数据
    else:
        num_worker


