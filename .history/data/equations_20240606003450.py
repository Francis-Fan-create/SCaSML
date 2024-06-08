import deepxde as dde
import numpy as np
import torch



class Equation(object):
    def __init__(self, n_input, n_output,have_exact_solution=False):
        self.n_input = n_input
        self.n_output = n_output
        self.have_exact_solution = have_exact_solution
    def PDE_Loss(self, x_t):
        #generator function in PDE
        raise NotImplementedError
    def Initial_Loss(self, x_t):
        #initial condition in the PDE
        raise NotImplementedError
    def Boundary_Loss(self, x_t):
        #boundary condition in the PDE
        raise NotImplementedError
    
    def mu(self, x_t):
        #drift coefficient in PDE
        raise NotImplementedError
    def sigma(self, x_t):
        #diffusion coefficient in PDE
        raise NotImplementedError

    def exact_solution(self, x_t):
        #exact solution of the PDE
        raise NotImplementedError
    def Data_Loss(self, x_t):
        #data loss in PDE
        if self.have_exact_solution:
            return self.exact_solution(x_t)
        
        raise NotImplementedError
    
    def geometry(self):
        #geometry of the domain
        raise NotImplementedError
    
class Explict_Solution_Example(Equation):
    def __init__(self, n_input, n_output, n_hidden, n_hidden_layers,have_exact_solution=True):
        super(Explict_Solution_Example, self).__init__(n_input, n_output, n_hidden, n_hidden_layers,have_exact_solution)
    def f(self, x_t):
        return 0.5 * x_t
    def g(self, x_t):
        return 0.5 * x_t
    def h(self, x_t):
        return 0.5 * x_t
    def mu(self, x_t):
        return 0
    def sigma(self, x_t):
        return 0.25
    def exact_solution(self, x_t):
        return 0.5 * x_t