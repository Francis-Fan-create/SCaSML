import deepxde as dde
import numpy as np
import torch


class Equation(object):
    '''Equation class for PDEs based on deepxde framework'''
    #all the vectors uses rows as index and columns as dimensions
    def __init__(self, n_input, n_output=1):
        #initialize the equation parameters
        #we assume that u is a scalar if do not specify n_output
        self.n_input = n_input #dimension of the input, including time
        self.n_output = n_output #dimension of the output
    def PDE_loss(self, x_t,u,z):
        #PINN loss in the PDE, used in ScaML to calculate epsilon
        raise NotImplementedError
    def gPDE_loss(self,x_t,u):
        #gPINN loss in the PDE, used for training
        raise NotImplementedError
    def terminal_constraint(self, x_t):
        #terminal constraint in the PDE
        raise NotImplementedError
    def boundary_constraint(self, x_t):
        #boundary constraint in the PDE
        raise NotImplementedError
    
    def mu(self, x_t):
        #drift coefficient in PDE, usually a vector
        raise NotImplementedError
    def sigma(self, x_t):
        #diffusion coefficient in PDE, usually a matrix
        raise NotImplementedError
    def f(self, x_t,u,z):
        #generator term in PDE, usually a vector
        #z is the gradient of u, usually a vector, since u is a scalar
        #note that z does not include the time dimension
        raise NotImplementedError
    def g(self,x_t):
        #terminal constraint in PDE, usually a vector
        if hasattr(self, 'terminal_constraint'):
            return self.terminal_constraint(x_t)
        else:
            raise NotImplementedError

    def exact_solution(self, x_t):
        #exact solution of the PDE, which will not be used in the training, but for testing
        raise NotImplementedError
    def data_loss(self, x_t):
        #data loss in PDE
        raise NotImplementedError
    
    def geometry(self,t0,T):
        #geometry of the domain
        raise NotImplementedError
    def terminal_condition(self,x_t,u):
        #terminal condition of the PDE, using hard constraint
        if hasattr(self, 'terminal_constraint') and hasattr(self, 'geometry'):
            return (x_t[:,-1] - self.T)*u+self.terminal_constraint(x_t) #need to be enforced on network class
        else:
            raise NotImplementedError
    def boundary_condition(self):
        #boundary condition of the PDE, using soft constraint
        if hasattr(self, 'boundary_constraint') and hasattr(self, 'geometry'):
            bc=dde.icbc.DirichletBC(self.geometry(),self.boundary_constraint, lambda _, on_boundary: on_boundary) #need to be enforced on generate_data method
            self.boundary_condition=bc
            return bc
        else:
            raise NotImplementedError
    
    def generate_data(self):
        #generate data for training
        raise NotImplementedError
    
class Explict_Solution_Example(Equation):
    '''Expamlpe of high dim PDE with exact solution'''
    def __init__(self, n_input, n_output=1):
        super(Explict_Solution_Example, self).__init__(n_input, n_output)
    def PDE_loss(self, x_t,u,z):
        n, d = x_t.shape
        hessian = torch.zeros(n, d, d, device=x_t.device, dtype=x_t.dtype)
    
        # Compute the gradient of the output with respect to the input
        output = net(x_t)[:, 0]
        grad_output = torch.ones(n, device=x_t.device, dtype=x_t.dtype)
        grad = torch.autograd.grad(output, x_t, grad_outputs=grad_output, create_graph=True)[0]
    
        for i in range(d-1):
            grad_i = grad[:, i]
            grad_grad_i = torch.autograd.grad(grad_i, x_t, grad_outputs=grad_output, create_graph=True)[0]
            hessian[:, i, :] = grad_grad_i
        return residual
    def gPDE_loss(self, x_t,u):
        #use gPINN loss in this example
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1)
        laplacian=0
        div=0
        for k in range(self.n_input-1): #here, we use a slower accumulating method to avoid computing more autograd, which is a tradeoff
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k) #lazy
            div += dde.grad.jacobian(u, x_t, i=0, j=k) #lazy
        residual=du_t + (self.sigma()**2 * u - 1/self.n_input - self.sigma()**2/2) * div + self.sigma()**2/2 * laplacian
        g_loss=[]
        for k in range(self.n_input-1):
            g_loss.append(0.01*dde.grad.jacobian(residual,x_t,i=0,j=k))
        g_loss.append(residual)
        return g_loss
    def terminal_constraint(self, x_t):
        result=np.exp(x_t[:,-1] + np.sum(x_t[0:self.n_input],axis=1)) / (1 + np.exp(x_t[:,-1] + np.sum(x_t[0:self.n_input],axis=1)))
        return result

    def mu(self, x_t=0):
        return 0
    def sigma(self, x_t=0):
        return 0.25
    def f(self, x_t,u,z):
        #generator term for this PDE
        div=np.sum(z[:,0:self.n_input])
        result=(self.sigma()**2 * u - 1/self.n_input - self.sigma()**2/2) * div
        return result
    
    def exact_solution(self, x_t):
        #exact solution of the example
        t = x_t[:, -1]
        x = x_t[:, :-1]
        sum_x = np.sum(x, axis=1)
        exp_term = np.exp(t + sum_x)
        result=1-1/(1+exp_term)
        return result
    
    def geometry(self,t0=0,T=0.5):
        #geometry of the domain, which is a hypercube
        self.t0=t0
        self.T=T
        spacedomain = dde.geometry.Hypercube([-0.5]*(self.n_input-1), [0.5]*(self.n_input-1)) 
        timedomain = dde.geometry.TimeDomain(t0, T) 
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain) #combine both domains
        return geom
    
    def generate_data(self, num_domain=7000,num_test=10):
        data = dde.data.TimePDE( #time dependent PDE
                                self.geometry(), #geometry of the boundary condition and terminal condition
                                self.gPDE_loss, #g_pde residual
                                [], #additional conditions other than PDE loss
                                num_domain=num_domain, #sample how many points in the domain
                                num_boundary=0, #sample how many points on the boundary
                                num_terminal=0,  #sample how many points for the terminal time
                                solution=self.exact_solution,   #incorporate authentic solution to evaluate error metrics
                                num_test=num_test #sample how many points for testing. If None, then the training point will be used.
                            )
        return data