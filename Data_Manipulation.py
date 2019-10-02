import torch
x = torch.empty(5, 3) #未初始化的Tensor
print (x)
x = torch.rand(5, 3)  #随机初始化的Tensor
print (x)
x = torch.zeros(5, 3, dtype = torch.long)
print (x)
x = torch.tensor([5.5, 3]) #直接根据数据创建
print (x)
x = x.new_ones(5, 3, dtype = torch.float64)
print (x)
x = torch.randn_like(x, dtype = torch.float)
print (x)

print (x.size())#等价,输出形状,返回tuple类型
print (x.shape)

y = torch.rand(5, 3)
print (x + y)
print (torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out = result)
print (result)

y.add_(x)
print (y)

y = x[0, :]#索引出一个
y += 1
print (y)
print (x[0, :])#索引出来的结果与元数据共享内存,修改一个,另一个也会修改

y = x.view(15)
z = x.view(-1, 5) #-1所指的维度可以根据其他维度的值推出来
print (x.size(), y.size(), z.size())
x += 1
print (x)
print (y) #view()共享内存,也加了1

x_cp = x.clone().view(15)#clone创造一个副本
x -= 1
print (x)
print (x_cp)

x = torch.randn(1)
print (x)
print (x.item())#将一个Tensor转换成一个python number

x = torch.arange(1, 3).view(1, 2)#广播机制
print (x)
y = torch.arange(1, 4).view(3, 1)
print (y)
print (x + y)

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x#索引,view不会新开内存,但是y = x + y 会新开内存
print (id(y) == id_before)# False

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x#通过[:]写进y对应的内存中
print (id(y) == id_before)#True

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out = y)# y += x, y.add_(x)
print (id(y) == id_before)

a = torch.ones(5)#使用numpy()将Tensor转换成Numpy数组
b = a.numpy()
print (a, b)

a += 1#共享内存
print (a, b)
b += 1
print (a, b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)#使用from_numpy()将Numpy数组转换成Tensor(cpu上除了CharTensor)
print (a, b)
a += 1
print (a, b)
b += 1
print (a, b)

c = torch.tensor(a)#该方法总是会进行数据拷贝,返回的Tensor和原来的数据不再共享内存
a += 1
print (a, c)

#方法to()可以将Tensor在cpu和GPU之间相互移动
#以上代码只有在Pytorch GPU版本上才会执行:
if torch.cuda.is_available():
    device = torch.device("cuda")#GPU
    y = torch.ones_like(x, device=device)#直接创建一个在GPU上的Tensor
    x = x.to(device)#等价于.to("cuda")
    z = x + y
    print (z)
    print (z.to("cpu", torch.double))#to()还可以同时更改数据类型
