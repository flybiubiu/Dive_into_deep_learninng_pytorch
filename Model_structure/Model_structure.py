import torch
from torch import nn

class MLP(nn.Module):
    #声明带有模型参数的层,这里声明了两个全连接层
    def __init__(self, **kwargs):#**kwargs表示关键字参数,它本质上是一个dict
        #调用MLP父类Block的构造函数来进行必要的初始化.这样在构造实例时还可以指定其他函数
        #参数,如"模型参数的访问,初始化和共享"一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        #super 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题
        self.hidden = nn.Linear(784, 256)#隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)#输出层

    #定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

X = torch.rand(2, 784)
net = MLP()
print (net)
print (net(X))

class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):#*args表示任何多个无名参数，它本质是一个tuple
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):#isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
            for key, module in args[0].items():
                self.add_module(key, module)#add_module方法会将module添加进self._modules(一个OrderedDict)
        else:#传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):#self._modules返回一个OrderedDict,保证会按照成员添加时的顺序遍历
        for module in self._modules.values():
            input = module(input)
        return input

net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print (net)
print (net(X))

net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))#类似List的append操作
print (net[-1])#类似List的索引访问
print (net)

net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU()
})
net['output'] = nn.Linear(256, 10)#添加
print (net['linear'])#访问
print (net.output)
print (net)

class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = torch.rand((20, 20), requires_grad = False)#不可训练参数(常数参数)
        self.linear = nn.Linear(20, 20)
    def forward(self, x):
        x = self.linear(x)#使用创建的常数参数,以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)
        #复用全连接层,等价于两个全连接层共享参数
        x = self.linear(x)
        #控制流,这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()

X = torch.rand(2, 20)
net = FancyMLP()
print (net)
print (net(X))

class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)

net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())

X = torch.rand(2, 40)#均匀分布
print(net)
print (net(X))