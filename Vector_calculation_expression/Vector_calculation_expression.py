import torch
from time import time

a = torch.ones(1000)
b = torch.ones(1000)

start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print (time() - start)

start = time()
d = a + b
print (time() - start)

a = torch.ones(3)
b = 10
print (a + b)