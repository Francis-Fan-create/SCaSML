import deepxde as dde
import numpy as np
import torch



class Equation(object):
    '''Equation class for PDEs based on deepxde framework'''
    def __init__(self, n_input, n_output,net,have_exact_solution=False):
        #initialize the equation parameters
        self.n_input = n_input #dimension of the input, including time
        self.n_output = n_output #dimension of the output
        self.have_exact_solution = have_exact_solution #whether the exact solution is known
        self.net = net #PINN network

    def PDE_Loss(self, x_t,u):
        #PINN loss in the PDE, can be a list as in gPINN
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
        #exact solution of the PDE, which will not be used in the training, but for testing
        raise NotImplementedError
    def Data_Loss(self, x_t):
        #data loss in PDE
        raise NotImplementedError
    
    def geometry(self):
        #geometry of the domain
        raise NotImplementedError
    
    def generate_data(self):
        #generate data for training
        raise NotImplementedError
    
class Explict_Solution_Example(Equation):
    '''Expamlpe of high dim PDE with exact solution'''
    def __init__(self, n_input, n_output, n_hidden, n_hidden_layers,have_exact_solution=True):
        super(Explict_Solution_Example, self).__init__(n_input, n_output, n_hidden, n_hidden_layers,have_exact_solution)

    def PDE_Loss(self, x_t,u):
        #use gPINN loss in this example
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1)
        laplacian=0
        div=0
        for k in range(self.n_input-1): #here, we use a slower accumulating method to avoid computing more autograd, which is a tradeoff
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k) #lazy
            div += dde.grad.jacobian(u, x_t, i=0, j=k) #lazy
        pde_loss=du_t + (self.sigma**2 * u - 1/self.n_input - self.sigma**2/2) * div + self.sigma**2/2 * laplacian
        g_loss=[]
        for k in range(self.n_input):
            g_loss.append(0.01*dde.grad.jacobian(pde_loss,x_t,i=0,j=k))
        g_loss.append(pde_loss)
        return g_loss
    def Initial_Loss(self, x_t):
        result=np.exp(0.5 + np.sum(x_t,axis=1)) / (1 + np.exp(0.5 + np.sum(x_t,axis=1)))
        return result

    def mu(self, x_t=0):
        return 0
    def sigma(self, x_t=0):
        return 0.25
    
    def exact_solution(self, x_t):
        #exact solution of the example
        s = x_t[:, -1]
        x = x_t[:, :-1]
        sum_x = np.sum(x, axis=1)
        exp_term = np.exp(s + sum_x)
        result=1-1/(1+exp_term)
        return result
    
    def geometry(self):
        #geometry of the domain, which is a hypercube
        spacedomain = dde.geometry.Hypercube([-0.5]*(self.n_input-1), [0.5]*(self.n_input-1)) 
        timedomain = dde.geometry.TimeDomain(0, 0.5) 
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain) #combine both domains
        return geom
    
    def generate_data(self)