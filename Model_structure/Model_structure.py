import torch
from torch import nn

class MLP(nn.Module):
    #声明带有模型参数的层,这里声明了两个全连接层
    def __init__(self, **kwargs):#**kwargs表示关键字参数,它本质上是一个dict
