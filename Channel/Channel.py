import torch
from torch import nn
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def corr2d_multi_in(X, K):
    #沿着X和K的第0维(通道维)分别计算再相加
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
print (corr2d_multi_in(X, K))

def corr2d_multi_in_out(X, K):
    #对K的第0维遍历,每次同输入X做相关计算,所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])

K = torch.stack([K, K + 1, K + 2])
print (K.shape) #torch.Size([3, 2, 2, 2])
print (corr2d_multi_in_out(X, K))

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)#全连接层的矩阵乘法
    return Y.view(c_o, h, w)

X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print ((Y1 -Y2).norm().item())

