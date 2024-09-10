import deepxde as dde
import numpy as np
import torch


import sys
import os
#add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from solvers.MLP import MLP # use MLP to deal with equations without explicit solutions

class Equation(object):
    '''Equation class for PDEs based on deepxde framework'''
    # all the vectors use rows as index and columns as dimensions
    
    def __init__(self, n_input, n_output=1):
        """
        Initialize the equation parameters.
        We assume that u is a scalar if n_output is not specified.
        
        Args:
            n_input (int): Dimension of the input, including time.
            n_output (int, optional): Dimension of the output. Defaults to 1.
        """
        self.n_input = n_input  # dimension of the input, including time
        self.n_output = n_output  # dimension of the output

    def PDE_loss(self, x_t, u, z):
        """
        PINN loss in the PDE, used in ScaSML to calculate epsilon.
        
        Args:
            x_t (tensor): The input data, shape (n_samples, n_input).
            u (tensor): The solution, shape (n_samples, n_output).
            z (tensor): The gradient of u w.r.t. x, shape (n_samples, n_input-1).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def gPDE_loss(self, x_t, u):
        """
        gPINN loss in the PDE, used for training.
        
        Args:
            x_t (tensor): The input data, shape (n_samples, n_input).
            u (tensor): The solution, shape (n_samples, n_output).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def terminal_constraint(self, x_t):
        """
        Terminal constraint in the PDE.
        
        Args:
            x_t (ndarray): The input data at terminal time, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
    
    def initial_constraint(self, x_t):
        """
        Initial constraint in the PDE.
        
        Args:
            x_t (ndarray): The input data at terminal time, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError    

    def Dirichlet_boundary_constraint(self, x_t):
        """
        Dirichlet boundary constraint in the PDE.
        
        Args:
            x_t (ndarray): The input data at the boundary, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def Neumann_boundary_constraint(self, x_t):
        """
        Neumann boundary constraint in the PDE.
        
        Args:
            x_t (ndarray): The input data at terminal time, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
    
    def mu(self, x_t):
        """
        Drift coefficient in PDE, usually a vector.
        
        Args:
            x_t (ndarray): The input data, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def sigma(self, x_t):
        """
        Diffusion coefficient in PDE, usually a matrix.
        
        Args:
            x_t (ndarray): The input data, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def f(self, x_t, u, z):
        """
        Generator term in PDE, usually a vector.
        z is the product of the gradient of u and sigma, usually a vector, since u is a scalar.
        Note that z does not include the time dimension.
        
        Args:
            x_t (ndarray): The input data, shape (n_samples, n_input).
            u (ndarray): The solution, shape (n_samples, n_output).
            z (ndarray): The gradient of u w.r.t. x, shape (n_samples, n_input-1).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def g(self, x_t):
        """
        Terminal constraint in PDE, usually a vector.
        
        Args:
            x_t (ndarray): The input data at terminal time, shape (n_samples, n_input).
            
        Returns:
            ndarray: The terminal constraint value, shape (n_samples, n_output).
            
        Raises:
            NotImplementedError: If the terminal_constraint method is not implemented.
        """
        if hasattr(self, 'terminal_constraint'):
            return self.terminal_constraint(x_t)
        else:
            raise NotImplementedError

    def exact_solution(self, x_t):
        """
        Exact solution of the PDE, which will not be used in the training, but for testing.
        
        Args:
            x_t (ndarray): The input data, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def data_loss(self, x_t):
        """
        Data loss in PDE.
        
        Args:
            x_t (ndarray): The input data, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
    
    def geometry(self, t0, T):
        """
        Geometry of the domain.
        
        Args:
            t0 (float): The initial time.
            T (float): The terminal time.
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
        
    def terminal_condition(self):
        """
        Terminal condition of the PDE, using hard constraint.
        
        Returns:
            dde.icbc.PointSetBC: The terminal condition boundary condition object.
            
        Raises:
            NotImplementedError: If the terminal_constraint or geometry method is not implemented.
        """
        if hasattr(self, 'terminal_constraint') and hasattr(self, 'geometry'):
            # use PointSetBC to enforce soft terminal condition
            # generate terminal point
            x = self.geomx.random_points(500)  # do not use uniform !!!
            t = self.T * np.ones((500, 1))
            my_data = np.concatenate((x, t), axis=1)
            self.my_data = my_data
            tc = dde.icbc.PointSetBC(my_data, self.terminal_constraint(my_data), 0)  # need to be enforced on generate_data method
            self.tc = tc
            return tc
        else:
            raise NotImplementedError

    def initial_condition(self):
        """
        Initial condition of the PDE, using soft constraint.
        
        Returns:
            dde.icbc.DirichletBC: The boundary condition object.
            
        Raises:
            NotImplementedError: If the boundary_constraint or geometry method is not implemented.
        """
        if hasattr(self, 'initial_constraint') and hasattr(self, 'geometry'):
            ic = dde.icbc.IC(self.geometry(), self.initial_constraint, lambda _, on_initial: on_initial)  # need to be enforced on generate_data method
            self.ic = ic
            return ic
        else:
            raise NotImplementedError

    def Dirichlet_boundary_condition(self):
        """
        Dirichlet boundary condition of the PDE, using soft constraint.
        
        Returns:
            dde.icbc.DirichletBC: The boundary condition object.
            
        Raises:
            NotImplementedError: If the boundary_constraint or geometry method is not implemented.
        """
        if hasattr(self, 'Dirichlet_boundary_constraint') and hasattr(self, 'geometry'):
            D_bc = dde.icbc.DirichletBC(self.geometry(), self.Dirichlet_boundary_constraint, lambda _, on_boundary: on_boundary)  # need to be enforced on generate_data method
            self.D_bc = D_bc
            return D_bc
        else:
            raise NotImplementedError

    def Neumann_boundary_condition(self):
        """
        Neumann boundary condition of the PDE, using soft constraint.
        
        Returns:
            dde.icbc.NeumannBC: The boundary condition object.
            
        Raises:
            NotImplementedError: If the boundary_constraint or geometry method is not implemented.
        """
        if hasattr(self, 'Neumann_boundary_constraint') and hasattr(self, 'geometry'):
            N_bc = dde.icbc.NeumannBC(self.geometry(), self.Neumann_boundary_constraint, lambda _, on_boundary: on_boundary)  # need to be enforced on generate_data method
            self.N_bc = N_bc
            return N_bc
        else:
            raise NotImplementedError
    
    def generate_data(self):
        """
        Generate data for training.
        
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
    
class Explicit_Solution_Example(Equation):
    '''
    Example of a high-dimensional PDE with an exact solution.
    '''
    def __init__(self, n_input, n_output=1):
        '''
        Initializes the PDE example with specified input and output dimensions.
        
        Parameters:
        - n_input (int): The dimension of the input space, including the time dimension.
        - n_output (int): The dimension of the output space. Defaults to 1.
        '''
        super().__init__(n_input, n_output)
    
    def PDE_loss(self, x_t,u,z):
        '''
        Calculates the PDE loss for given inputs.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        - z (tensor): Tensor of shape (batch_size, n_input-1), representing gradients of u w.r.t. x.
        
        Returns:
        - residual (tensor): The residual of the PDE of shape (batch_size, n_output).
        '''
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1) # Computes the time derivative of u.
        laplacian=0
        div=0
        for k in range(self.n_input-1): # Accumulates laplacian and divergence over spatial dimensions.
            laplacian +=dde.grad.jacobian(z, x_t, i=k, j=k) # Computes the laplacian of z.
            div += dde.grad.jacobian(u, x_t, i=0, j=k) # Computes the divergence of u.
        residual=du_t + (self.sigma()**2 * u - 1/(self.n_input-1) - self.sigma()**2/2) * div+ self.sigma()**2/2 * laplacian
        return residual 
    
    def gPDE_loss(self, x_t,u):
        '''
        Calculates the generalized PDE loss using the gPINN approach.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        
        Returns:
        - g_loss (list of tensor): List of gradients of the residual for gPINN loss, with the last element being the residual itself.
        '''
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1) # Computes the time derivative of u.
        laplacian=0
        div=0
        for k in range(self.n_input-1): # Accumulates laplacian and divergence over spatial dimensions.
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k) # Computes the laplacian of u.
            div += dde.grad.jacobian(u, x_t, i=0, j=k) # Computes the divergence of u.
        residual=du_t + (self.sigma()**2 * u - 1/(self.n_input-1) - self.sigma()**2/2) * div + self.sigma()**2/2 * laplacian
        g_loss=[]
        for k in range(self.n_input-1): # Accumulates gradients of the residual for gPINN loss.
            g_loss.append(0.01*dde.grad.jacobian(residual,x_t,i=0,j=k)) # Computes gradient penalty.
        g_loss.append(residual) # Adds the residual to the loss.
        return g_loss
    
    def terminal_constraint(self, x_t):
        '''
        Defines the terminal constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 1D tensor of shape (batch_size,), representing the terminal constraint.
        '''
        result= 1-1 / (1 + np.exp(x_t[:,-1] + np.sum(x_t[:,:self.n_input-1],axis=1))) # Computes the terminal constraint.
        return result 

    def mu(self, x_t=0):
        '''
        Returns the drift coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The drift coefficient.
        '''
        return 0
    
    def sigma(self, x_t=0):
        '''
        Returns the diffusion coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The diffusion coefficient.
        '''
        return 0.25
    
    def f(self, x_t,u,z):
        '''
        Defines the generator term for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (ndarray): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        - z (ndarray): Tensor of shape (batch_size, n_input-1), representing gradients.
        
        Returns:
        - result (ndarray): A 2D array of shape (batch_size, n_output), representing the generator term.
        '''
        # div=np.sum(z,axis=1) # Computes the divergence of z.
        # result=(self.sigma()**2 * u - 1/(self.n_input-1) - self.sigma()**2/2) * div[:,np.newaxis] # Computes the generator term.
        dim=self.n_input-1
        result=self.sigma() * (u - (2+self.sigma() * self.sigma() * dim) / (2 * self.sigma() * self.sigma() *dim)) * np.sum(z, axis=1, keepdims=True)
        return result
    
    def exact_solution(self, x_t):
        '''
        Computes the exact solution of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An arrary of shape (batch_size, n_output), representing the exact solution.
        '''
        t = x_t[:, -1]
        x = x_t[:, :-1]
        sum_x = np.sum(x, axis=1)
        exp_term =np.exp(t + sum_x) # Computes the exponential term of the solution.
        result=1-1/(1+exp_term) # Computes the exact solution.
        return result
    
    def geometry(self,t0=0,T=0.5):
        '''
        Defines the geometry of the domain for the PDE.
        
        Parameters:
        - t0 (float): Initial time.
        - T (float): Terminal time.
        
        Returns:
        - geom (dde.geometry.GeometryXTime): A GeometryXTime object representing the domain.
        '''
        self.t0=t0
        self.T=T
        spacedomain = dde.geometry.Hypercube([-0.5]*(self.n_input-1), [0.5]*(self.n_input-1)) # Defines the spatial domain, for train
        # spacedomain = dde.geometry.Hypercube([-0.1]*(self.n_input-1), [0.1]*(self.n_input-1)) # Defines the spatial domain , for test
        timedomain = dde.geometry.TimeDomain(t0, T) # Defines the time domain.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain) # Combines spatial and time domains.
        self.geomx=spacedomain
        self.geomt=timedomain
        return geom
    
    def generate_data(self, num_domain=100):
        '''
        Generates data for training the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        
        Returns:
        - data (dde.data.TimePDE): A TimePDE object containing the training data.
        '''
        geom=self.geometry() # Defines the geometry of the domain.
        self.terminal_condition() # Generates terminal condition.
        data = dde.data.TimePDE(
                                geom, # Geometry of the domain.
                                self.gPDE_loss, # gPDE loss function.
                                [self.tc], # Additional conditions.
                                num_domain=num_domain, # Number of domain points.
                                num_boundary=0, # Number of boundary points.
                                num_initial=0,  # Number of initial points.
                                anchors=self.my_data, # Enforces terminal points.
                                solution=self.exact_solution,   # Incorporates exact solution for error metrics.
                                num_test=None # Number of test points.
                            )
        return data
    
class Complicated_HJB(Equation):
    '''
    Complicated HJB equation.
    '''
    def __init__(self, n_input, n_output=1):
        '''
        Initializes the PDE example with specified input and output dimensions.
        
        Parameters:
        - n_input (int): The dimension of the input space, including the time dimension.
        - n_output (int): The dimension of the output space. Defaults to 1.
        '''
        super().__init__(n_input, n_output)
    
    def PDE_loss(self, x_t,u,z):
        '''
        Calculates the PDE loss for given inputs.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        - z (tensor): Tensor of shape (batch_size, n_input-1), representing gradients of u w.r.t. x.
        
        Returns:
        - residual (tensor): The residual of the PDE of shape (batch_size, n_output).
        '''
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1) # Computes the time derivative of u.
        laplacian=0
        div=0
        dim=self.n_input-1
        for k in range(self.n_input-1): # Accumulates laplacian and divergence over spatial dimensions.
            laplacian +=dde.grad.jacobian(z, x_t, i=k, j=k) # Computes the laplacian of z.
            div += dde.grad.jacobian(u, x_t, i=0, j=k) # Computes the divergence of u.
        residual=du_t -(1/dim)*div+2+laplacian # Computes the residual of the PDE.
        return residual 
    
    def gPDE_loss(self, x_t,u):
        '''
        Calculates the generalized PDE loss using the gPINN approach.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        
        Returns:
        - g_loss (list of tensor): List of gradients of the residual for gPINN loss, with the last element being the residual itself.
        '''
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1) # Computes the time derivative of u.
        laplacian=0
        div=0
        dim=self.n_input-1
        for k in range(self.n_input-1): # Accumulates laplacian and divergence over spatial dimensions.
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k) # Computes the laplacian of u.
            div += dde.grad.jacobian(u, x_t, i=0, j=k) # Computes the divergence of u.
        residual=du_t -(1/dim)*div+2+laplacian
        g_loss=[]
        for k in range(self.n_input-1): # Accumulates gradients of the residual for gPINN loss.
            g_loss.append(0.01*dde.grad.jacobian(residual,x_t,i=0,j=k)) # Computes gradient penalty.
        g_loss.append(residual) # Adds the residual to the loss.
        return g_loss
    
    def terminal_constraint(self, x_t):
        '''
        Defines the terminal constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 1D tensor of shape (batch_size,), representing the terminal constraint.
        '''
        x=x_t[:,:self.n_input-1] # Extracts the spatial coordinates.
        sum_x=np.sum(x,axis=1) # Computes the sum of spatial coordinates.
        result=sum_x # Computes the terminal constraint.
        return result 

    def mu(self, x_t=0):
        '''
        Returns the drift coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The drift coefficient.
        '''
        dim=self.n_input-1
        return -1/dim
    
    def sigma(self, x_t=0):
        '''
        Returns the diffusion coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The diffusion coefficient.
        '''
        return np.sqrt(2)
    
    def f(self, x_t,u,z):
        '''
        Defines the generator term for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (ndarray): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        - z (ndarray): Tensor of shape (batch_size, n_input-1), representing gradients.
        
        Returns:
        - result (ndarray): A 2D array of shape (batch_size, n_output), representing the generator term.
        '''
        return 2*np.ones_like(u)
    
    def exact_solution(self, x_t):
        '''
        Computes the exact solution of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An arrary of shape (batch_size, ), representing the exact solution.
        '''
        t = x_t[:, -1]
        x = x_t[:, :-1]
        sum_x = np.sum(x, axis=1)
        result=sum_x+(self.T-t)
        return result
    
    def geometry(self,t0=0,T=0.5):
        '''
        Defines the geometry of the domain for the PDE.
        
        Parameters:
        - t0 (float): Initial time.
        - T (float): Terminal time.
        
        Returns:
        - geom (dde.geometry.GeometryXTime): A GeometryXTime object representing the domain.
        '''
        self.t0=t0
        self.T=T
        spacedomain = dde.geometry.Hypercube([-0.5]*(self.n_input-1), [0.5]*(self.n_input-1)) # Defines the spatial domain, for train
        # spacedomain = dde.geometry.Hypercube([-0.1]*(self.n_input-1), [0.1]*(self.n_input-1)) # Defines the spatial domain , for test
        timedomain = dde.geometry.TimeDomain(t0, T) # Defines the time domain.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain) # Combines spatial and time domains.
        self.geomx=spacedomain
        self.geomt=timedomain
        return geom
    
    def generate_data(self, num_domain=100):
        '''
        Generates data for training the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        
        Returns:
        - data (dde.data.TimePDE): A TimePDE object containing the training data.
        '''
        geom=self.geometry() # Defines the geometry of the domain.
        self.terminal_condition() # Generates terminal condition.
        data = dde.data.TimePDE(
                                geom, # Geometry of the domain.
                                self.gPDE_loss, # gPDE loss function.
                                [self.tc], # Additional conditions.
                                num_domain=num_domain, # Number of domain points.
                                num_boundary=0, # Number of boundary points.
                                num_initial=0,  # Number of initial points.
                                anchors=self.my_data, # Enforces terminal points.
                                solution=self.exact_solution,   # Incorporates exact solution for error metrics.
                                num_test=None # Number of test points.
                            )
        return data

class Neumann_Boundary(Equation):
    '''
    An Equation with Neumann boundary condition.
    '''
    def __init__(self, n_input, n_output=1):
        '''
        Initializes the PDE example with specified input and output dimensions.
        
        Parameters:
        - n_input (int): The dimension of the input space, including the time dimension.
        - n_output (int): The dimension of the output space. Defaults to 1.
        '''
        super().__init__(n_input, n_output)
    
    def PDE_loss(self, x_t,u,z):
        '''
        Calculates the PDE loss for given inputs.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        - z (tensor): Tensor of shape (batch_size, n_input-1), representing gradients of u w.r.t. x.
        
        Returns:
        - residual (tensor): The residual of the PDE of shape (batch_size, n_output).
        '''
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1) # Computes the time derivative of u.
        x_1=x_t[:,0,None]
        x_2=x_t[:,1,None]
        laplacian=0
        for k in range(self.n_input-1): # Accumulates laplacian and divergence over spatial dimensions.
            laplacian +=dde.grad.jacobian(z, x_t, i=k, j=k) # Computes the laplacian of z.
        residual=du_t +(2*u-(torch.pi**2/2+2)*torch.sin(torch.pi/2*x_1)*torch.cos(torch.pi/2*x_2))-laplacian # Computes the residual of the PDE.
        return residual 
    
    def gPDE_loss(self, x_t,u):
        '''
        Calculates the generalized PDE loss using the gPINN approach.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        
        Returns:
        - g_loss (list of tensor): List of gradients of the residual for gPINN loss, with the last element being the residual itself.
        '''
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1) # Computes the time derivative of u.
        x_1=x_t[:,0,None]
        x_2=x_t[:,1,None]
        laplacian=0
        for k in range(self.n_input-1): # Accumulates laplacian and divergence over spatial dimensions.
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k) # Computes the laplacian of z.
        residual=du_t +(2*u-(torch.pi**2/2+2)*torch.sin(torch.pi/2*x_1)*torch.cos(torch.pi/2*x_2))-laplacian # Computes the residual of the PDE.
        g_loss=[]
        for k in range(self.n_input-1): # Accumulates gradients of the residual for gPINN loss.
            g_loss.append(0.01*dde.grad.jacobian(residual,x_t,i=0,j=k)) # Computes gradient penalty.
        g_loss.append(residual) # Adds the residual to the loss.
        return g_loss

    def terminal_constraint(self, x_t):
        '''
        Computes the terminal constraint of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An arrary of shape (batch_size, ), representing the exact solution.
        '''
        x_1=x_t[:,0] # Extracts the spatial coordinate.
        x_2=x_t[:,1] # Extracts the spatial coordinate.
        result=np.sin(np.pi/2*x_1)*np.cos(np.pi/2*x_2) # Computes the exact solution.
        return result
    
    def Neumann_boundary_constraint(self, x_t):
        '''
        Defines the Neumann boundary constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 1D tensor of shape (batch_size,), representing the terminal constraint.
        '''
        x=x_t[:,:self.n_input-1] # Extracts the spatial coordinates.
        boundary_normal=self.geomx.boundary_normal(x) # Computes the normal vector to the boundary.
        dot_tensor=np.zeros_like(boundary_normal) # Initializes the dot product tensor.
        dot_tensor[:,0]=np.pi/2*np.cos(np.pi/2*x[:,0])*np.cos(np.pi/2*x[:,1]) # Computes the dot product.
        dot_tensor[:,1]=np.pi/2*np.sin(np.pi/2*x[:,0])*-np.sin(np.pi/2*x[:,1]) # Computes the dot product.
        result=np.sum(boundary_normal*dot_tensor,axis=1)
        return result     

    def mu(self, x_t=0):
        '''
        Returns the drift coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The drift coefficient.
        '''
        return 0
    
    def sigma(self, x_t=0):
        '''
        Returns the diffusion coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The diffusion coefficient.
        '''
        return -np.sqrt(2)
    
    def f(self, x_t,u,z):
        '''
        Defines the generator term for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (ndarray): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        - z (ndarray): Tensor of shape (batch_size, n_input-1), representing gradients.
        
        Returns:
        - result (ndarray): A 2D array of shape (batch_size, n_output), representing the generator term.
        '''
        x_1=x_t[:,0,np.newaxis]
        x_2=x_t[:,1,np.newaxis]
        result=2*u-(np.pi**2/2+2)*np.sin(np.pi/2*x_1)*np.cos(np.pi/2*x_2)
        return result
    
    def exact_solution(self, x_t):
        '''
        Computes the exact solution of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An arrary of shape (batch_size, ), representing the exact solution.
        '''
        x_1=x_t[:,0] # Extracts the spatial coordinate.
        x_2=x_t[:,1] # Extracts the spatial coordinate.
        result=np.sin(np.pi/2*x_1)*np.cos(np.pi/2*x_2) # Computes the exact solution.
        return result 
    
    def geometry(self,t0=0,T=0.5):
        '''
        Defines the geometry of the domain for the PDE.
        
        Parameters:
        - t0 (float): Initial time.
        - T (float): Terminal time.
        
        Returns:
        - geom (dde.geometry.GeometryXTime): A GeometryXTime object representing the domain.
        '''
        self.t0=t0
        self.T=T
        spacedomain = dde.geometry.Hypercube([-0.5]*(self.n_input-1), [0.5]*(self.n_input-1)) # Defines the spatial domain, for train
        # spacedomain = dde.geometry.Hypercube([-0.1]*(self.n_input-1), [0.1]*(self.n_input-1)) # Defines the spatial domain , for test
        timedomain = dde.geometry.TimeDomain(t0, T) # Defines the time domain.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain) # Combines spatial and time domains.
        self.geomx=spacedomain
        self.geomt=timedomain
        return geom
    
    def generate_data(self, num_domain=100,num_N_boundary=100):
        '''
        Generates data for training the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        - num_N_boundary (int): Number of points to sample on the Neumann boundary.
        
        Returns:
        - data (dde.data.TimePDE): A TimePDE object containing the training data.
        '''
        geom=self.geometry() # Defines the geometry of the domain.
        self.Neumann_boundary_condition()  # Generates boundary condition.
        data = dde.data.TimePDE(
                                geom, # Geometry of the domain.
                                self.gPDE_loss, # gPDE loss function.
                                [self.N_bc], # Additional conditions.
                                num_domain=num_domain, # Number of domain points.
                                num_boundary=num_N_boundary, # Number of boundary points.
                                solution=self.exact_solution,   # Incorporates exact solution for error metrics.
                                num_test=None # Number of test points.
                            )
        return data