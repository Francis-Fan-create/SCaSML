import deepxde as dde
import numpy as np
import torch



class Equation(object):
    '''Equation class for PDEs based on deepxde framework'''
    def __init__(self, n_input, n_output,net,have_exact_solution=False):
        #initialize the equation parameters
        self.n_input = n_input
        self.n_output = n_output
        self.have_exact_solution = have_exact_solution
        self.net = net
    def PDE_Loss(self, x_t,u):
        #PINN loss in the PDE
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
            return torch.mean((self.net(x_t) - self.exact_solution(x_t)) ** 2)
        else:
            raise NotImplementedError
    
    def geometry(self):
        #geometry of the domain
        raise NotImplementedError
    
class Explict_Solution_Example(Equation):
    def __init__(self, n_input, n_output, n_hidden, n_hidden_layers,have_exact_solution=True):
        super(Explict_Solution_Example, self).__init__(n_input, n_output, n_hidden, n_hidden_layers,have_exact_solution)
    def mu(self, x_t=0):
        return 0
    def sigma(self, x_t=0):
        return 0.25
    def PDE_Loss(self, x_t,u):
        #use gPINN loss in this example
        du_t = dde.grad.jacobian(u,x_t,i=0,j=100)
        laplacian=0
        div=0
        for k in range(d): #here, we use a slower accumulating method to avoid computing more autograd, which is a tradeoff
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k) #lazy
            div += dde.grad.jacobian(u, x_t, i=0, j=k) #lazy
        pde_loss=du_t + (self.sigma**2 * u - 1/d - self.sigma**2/2) * div + sigma**2/2 * laplacian
        g_loss=[]
        for k in range(d):
            g_loss.append(0.01*dde.grad.jacobian(pde_loss,x_t,i=0,j=k))
        # print('g_loss:',k)
        g_loss.append(pde_loss)
        return g_loss