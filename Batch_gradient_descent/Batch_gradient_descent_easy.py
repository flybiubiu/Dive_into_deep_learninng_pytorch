import numpy as np
import time
import torch
from torch import nn, optim
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
torch.cuda.set_device(2)
print (torch.cuda.current_device())

def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams, features, labels, batch_size = 10, num_epochs = 2):
    #初始化模型
    net = nn.Sequential(nn.Linear(features.shape[-1], 1))
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)
    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2
    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size, shuffle = True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            #除以2是为了和train_ch7保持一致,因为squared_loss中除了2
            l = loss(net(X).view(-1), y) / 2
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print ('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.savefig('0_05.jpg')
    d2l.plt.cla()

features, labels = d2l.get_data_ch7()
print (features.shape)
train_pytorch_ch7(optim.SGD, {"lr": 0.05}, features, labels, 10)