import torch
import torch.nn as nn
import numpy as np
import d2lzh_pytorch as d2l

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
features = torch.randn(())

def fit_and_plot_pytorch(wd):
    # 对权重参数衰减,权重名称一般是以weight结尾
    net = nn.Linear(num_inputs, 1)
