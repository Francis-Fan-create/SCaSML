import deepxde as dde
import numpy as np
import torch
from deepxde.backend import tensorflow as tf
from torch import nn
from torch.nn import functional as F

'''Initialization'''
# set random seed of deepxde
dde.config.set_random_seed(1234)
# fix random seed for numpy
np.random.seed(1234)
# Set default dtype in PyTorch to float64
torch.set_default_dtype(torch.float64)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name())

class Equation(object):
    def __init__(self, n_input, n_output, n_hidden, n_hidden_layers):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers

    def f(self, x, y):
        raise NotImplementedError

    def g(self, x):
        raise NotImplementedError

    def h(self, x):
        raise NotImplementedError

    def loss(self, x, y):
        return torch.mean((self.f(x, y))**2)

    def loss_bc(self, x, y):
        return torch.mean((self.g(x) - y)**2)

    def loss_ic(self, x, y):
        return torch.mean((self.h(x) - y)**2)