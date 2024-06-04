import deepxde as dde
import numpy as np
import torch



class Equation(object):
    def __init__(self, n_input, n_output, n_hidden, n_hidden_layers,have_exact_solution=False):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        self.have_exact_solution = have_exact_solution
    def f(self, x_t):
        #nonlinear term in the PDE
        raise NotImplementedError

    def g(self, x)):
        #initial condition in the PDE
        raise NotImplementedError

    def h(self, x):
        #boundary condition in the PDE
        raise NotImplementedError

    def exact_solution(self, x):
        #exact solution of the PDE
        raise NotImplementedError