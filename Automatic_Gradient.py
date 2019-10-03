import torch
x = torch.ones(2, 2, requires_grad = True)#叶子节点
print (x)
print (x.grad_fn)#每个Tensor都有一个.grad_fn属性,该属性即创建该Tensor的Function
#就是说该Tensor是不是通过某些运算得到的,若是,则grad_fn返回一个与这些运算相关的对象,否则是None

y = x + 2
print (y)
print (y.grad_fn)
print (x.is_leaf, y.is_leaf)#判断是否为叶子节点

z = y * y * 3
out = z.mean()
print (z, out)#out是一个标量

a = torch.randn(2, 2)# 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print (a.requires_grad)#False
a.requires_grad_(True)
print (a.requires_grad)#True
b = (a * a).sum()
print (b.grad_fn)
out.backward()#等价于out.backward(torch.tensor(1.))
print (x.grad)

#再来反向传播一次,注意grad是累加的
out2 = x.sum()
out2.backward()
print (x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print (x.grad)

x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad = True)
y = 2 * x
z = y.view(2, 2)
print (z)
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype = torch.float)
z.backward(v)
print (x.grad)

x = torch.tensor(1.0, requires_grad = True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
print (x.requires_grad)
print (y1, y1.requires_grad)#True
print (y2, y2.requires_grad)#False
print (y3, y3.requires_grad)#True
y3.backward()
print (x.grad)

x = torch.ones(1, requires_grad = True)
print (x.data)#还是一个tensor
print (x.data.requires_grad)#但是已经是独立于计算图之外
y = 2 * x
x.data *= 100
y.backward()
print (x)
print (x.grad)
