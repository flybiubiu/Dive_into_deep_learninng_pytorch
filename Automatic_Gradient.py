import torch
x = torch.ones(3, 3, requires_grad = True)
print (x)
print (x.grad_fn)#每个