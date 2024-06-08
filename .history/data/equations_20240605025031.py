import deepxde as dde
import numpy as np
import torch



class Equation(object):
    def __init__(self, n_input, n_output, n_hidden, n_hidden_layers):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers

    def f(self, x, y):
        #nonlinear term in the PDE
        raise NotImplementedError

    def g(self, x):
        #initial
        raise NotImplementedError

    def h(self, x):
        raise NotImplementedError

    def loss(self, x, y):
        return torch.mean((self.f(x, y))**2)

    def loss_bc(self, x, y):
        return torch.mean((self.g(x) - y)**2)

    def loss_ic(self, x, y):
        return torch.mean((self.h(x) - y)**2)