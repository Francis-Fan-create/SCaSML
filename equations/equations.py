import deepxde as dde
import numpy as np
import torch
import sys
import os
from jax import jit, random, vmap
from functools import partial
import jax.numpy as jnp
#add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

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

    def grad(self, x_t,u):
        '''
        Calculates the gradient for given inputs.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        
        Returns:
        - div (tensor): The div of the PDE of shape (batch_size, n_output).
        '''
        gradient = jnp.zeros((x_t.shape[0],self.n_input-1))
        for k in range(self.n_input-1): # Accumulates laplacian and divergence over spatial dimensions.
            gradient=gradient.at[:,k].set(dde.grad.jacobian(u, x_t, i=0, j=k)[0].flatten()) # Computes the divergence of u.
        return gradient

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

    def data_loss(self):
        """
        Data loss in PDE.
        
        Args:
            x_t (ndarray): The input data, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError if exact_solution is not implemented.
        """
        if hasattr(self, 'exact_solution') and hasattr(self, 'geometry'):
            # use PointSetBC to enforce soft terminal condition
            # generate terminal point
            geom = self.geometry()
            my_data = geom.random_points(160)  # do not use uniform !!!
            dlc = dde.icbc.PointSetBC(my_data, self.exact_solution(my_data), 0)  # need to be enforced on generate_data method
            self.dlc = dlc
            return dlc
        else:
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
        Terminal condition of the PDE, using soft constraint.
        
        Returns:
            dde.icbc.PointSetBC: The terminal condition boundary condition object.
            
        Raises:
            NotImplementedError: If the terminal_constraint or geometry method is not implemented.
        """
        if hasattr(self, 'terminal_constraint') and hasattr(self, 'geometry'):
            # use PointSetBC to enforce soft terminal condition
            # generate terminal point
            x =  random.normal(random.PRNGKey(0),(500,self.n_input-1)) # do not use uniform !!!
            t = self.T * jnp.ones((500, 1))
            my_data = jnp.concatenate((x, t), axis=1)
            tc = dde.icbc.PointSetBC(my_data, self.terminal_constraint(my_data), 0)  # need to be enforced on generate_data method
            self.tc = tc
            return tc
        else:
            raise NotImplementedError
    
    def terminal_transform(self,x_t,u):
        '''
        Hard Terminal condition in the PDE.

        Args:
            x_t (ndarray): The input data at terminal time, shape (n_samples, n_input).
            u (ndarray): The output data of PINN, shape (n_samples, n_output).
        
        Returns:
            result (ndarray): A 1D tensor of shape (n_samples,), representing the terminal constraint.

        Raises:
            NotImplementedError: If the terminal_constraint or geometry method is not implemented.
        '''
        if hasattr(self, 'terminal_constraint') and hasattr(self, 'geometry'):
            t = x_t[:, -1]
            T = self.T
            output = ((T-t[:,jnp.newaxis])/T)*u + (t[:,jnp.newaxis]/T)*self.terminal_constraint(x_t)
            return output   
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
        
    def test_geometry(self, t0=0, T=0.5):
        '''
        Defines the geometry of the domain for the PDE.
        
        Parameters:
        - t0 (float): Initial time.
        - T (float): Terminal time.
        
        Returns:
        - geom (dde.geometry.GeometryXTime): A GeometryXTime object representing the domain.
        '''
        self.t0 = t0
        self.test_T = T
        self.test_radius = 0.5
        spacedomain = dde.geometry.Hypercube([-self.test_radius] * (self.n_input - 1), [self.test_radius] * (self.n_input - 1))  # Defines the spatial domain, for test.
        timedomain = dde.geometry.TimeDomain(t0, self.test_T)  # Defines the time domain for test.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain)  # Combines spatial and time domains.
        self.test_geomx = spacedomain
        self.test_geomt = timedomain
        return geom  
    
    def generate_test_data(self, num_domain=100, num_boundary=20):
        '''
        Generates data for testing the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        - num_boundary (int): Number of points to sample on the boundary.
        
        Returns:
        - data (tuple): A tuple containing domain points and boundary points.
        '''
        geom = self.test_geometry()  # Defines the geometry of the domain.
        data1 = geom.random_points(num_domain)  # Generates random points in the domain.
        data2 = geom.random_boundary_points(num_boundary)  # Generates random points on the boundary.
        return data1, data2
    
    def generate_data(self):
        """
        Generate data for training.
        
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
    
class Grad_Dependent_Nonlinear(Equation):
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
        self.uncertainty = 1e-2
        self.norm_estimation = 1
    
    def PDE_loss(self, x_t,u):
        '''
        Calculates the PDE loss for given inputs.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        
        Returns:
        - residual (tensor): The residual of the PDE of shape (batch_size, n_output).
        '''
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1)[0] # Computes the time derivative of u.
        laplacian=0
        div=0
        dim = self.n_input-1
        MC = int(self.n_input/4)
        # randomly choose MC dims to compute hessian and div
        idx_list = np.random.choice(self.n_input-1, MC, replace=False)
        for k in idx_list: # Accumulates laplacian and divergence over spatial dimensions.
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k)[0] # Computes the laplacian of z.
            div += dde.grad.jacobian(u, x_t, i=0, j=k)[0] # Computes the divergence of u.
        laplacian *= dim/MC
        div *= dim/MC
        residual=du_t + (self.sigma()**2 * u[0] - 1/(self.n_input-1) - self.sigma()**2/2) * div + self.sigma()**2/2* laplacian
        return residual

    @partial(jit,static_argnames=["self"])
    def initial_constraint(self, x_t):
        '''
        Computes the initial contraint of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An arrary of shape (batch_size, n_output), representing the initial constraint.
        '''
        x = x_t[:, :-1]
        sum_x = jnp.sum(x, axis=1)
        exp_term = jnp.exp(sum_x)  # Computes the exponential term of the solution.
        result = 1 - 1 / (1 + exp_term)  # Computes the exact solution.
        result = result[:, jnp.newaxis]  # Convert to 2D
        return result

    @partial(jit,static_argnames=["self"])
    def terminal_constraint(self, x_t):
        '''
        Defines the terminal constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 1D tensor of shape (batch_size,), representing the terminal constraint.
        '''
        x = x_t[:, :-1]
        sum_x = jnp.sum(x, axis=1)
        exp_term = jnp.exp(self.T+sum_x)  # Computes the exponential term of the solution.
        result = 1 - 1 / (1 + exp_term)  # Computes the exact solution.
        result = result[:, jnp.newaxis]  # Convert to 2D
        return result
    
    @partial(jit,static_argnames=["self"])
    def Dirichlet_boundary_constraint(self, x_t):
        '''
        Defines the Dirichlet boundary constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 1D tensor of shape (batch_size,), representing the Dirichlet boundary constraint.
        '''
        t = x_t[:, -1]
        x = x_t[:, :-1]
        sum_x = jnp.sum(x, axis=1)
        exp_term = jnp.exp(t + sum_x)  # Computes the exponential term of the solution.
        result = 1 - 1 / (1 + exp_term)  # Computes the exact solution.
        result = result[:, jnp.newaxis]  # Convert to 2D
        return result

    def mu(self, x_t=0):
        '''
        Returns the drift coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The drift coefficient.
        '''
        d = self.n_input-1
        sigma = self.sigma()
        result = -1/d - sigma**2/ 2
        return result
    
    def sigma(self, x_t=0):
        '''
        Returns the diffusion coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The diffusion coefficient.
        '''
        return 0.25
    
    @partial(jit,static_argnames=["self"])
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
        result= self.sigma() * u * jnp.sum(z, axis=1, keepdims=True)
        return result
    
    @partial(jit,static_argnames=["self"])
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
        sum_x = jnp.sum(x, axis=1)
        exp_term = jnp.exp(t + sum_x)  # Computes the exponential term of the solution.
        result = 1 - 1 / (1 + exp_term)  # Computes the exact solution.
        result = result[:, jnp.newaxis]  # Convert to 2D
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
        timedomain = dde.geometry.TimeDomain(t0, T) # Defines the time domain.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain) # Combines spatial and time domains.
        self.geomx=spacedomain
        self.geomt=timedomain
        return geom
    
    def generate_data(self, num_domain=2500):
        '''
        Generates data for training the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        
        Returns:
        - data (dde.data.TimePDE): A TimePDE object containing the training data.
        '''
        geom=self.geometry() # Defines the geometry of the domain.
        # self.terminal_condition() # Generates terminal condition.
        self.Dirichlet_boundary_condition() # Generates Dirichlet boundary condition.
        self.initial_condition() # Generate initial condition
        # self.data_loss() # Generates data loss. 
        data = dde.data.TimePDE(
                                geom, # Geometry of the domain.
                                self.PDE_loss, # PDE loss function.
                                [self.ic, self.D_bc], # Additional conditions.
                                num_domain=num_domain, # Number of domain points.
                                num_boundary=100, # Number of boundary points.
                                num_initial=160,  # Number of initial points.
                                solution=self.exact_solution   # Incorporates exact solution for error metrics.
                            )
        return data

class Diffusion_Reaction(Equation):
    '''
    Diffusion Reaction equation.
    '''
    def __init__(self, n_input, n_output=1):
        '''
        Initializes the PDE example with specified input and output dimensions.
        
        Parameters:
        - n_input (int): The dimension of the input space, including the time dimension.
        - n_output (int): The dimension of the output space. Defaults to 1.
        '''
        super().__init__(n_input, n_output)
        self.uncertainty = 1e-2
        self.norm_estimation = jnp.sqrt(jnp.e)
    
    def PDE_loss(self, x_t,u):
        '''
        Calculates the PDE loss for given inputs.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        
        Returns:
        - residual (tensor): The residual of the PDE of shape (batch_size, n_output).
        '''
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1)[0] # Computes the time derivative of u.
        laplacian=0
        MC = int(self.n_input/4)
        dim=self.n_input-1
        # randomly choose MC dims to compute hessian and div
        idx_list = np.random.choice(self.n_input-1, MC, replace=False)
        for k in idx_list: # Accumulates laplacian and divergence over spatial dimensions.
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k)[0] # Computes the laplacian of z.
        laplacian *= dim/MC
        x_1 = x_t[:, 0]
        x_2 = x_t[:, 1]
        t = x_t[:, -1]
        f_val = (jnp.pi**2-2)*jnp.sin(jnp.pi*x_1/2)*jnp.cos(jnp.pi*x_2/2)*jnp.exp(-t) - 4* (jnp.sin(jnp.pi*x_1/2)**2)*jnp.cos(jnp.pi*x_2/2)*jnp.exp(-2*t)
        f_val = f_val[:,jnp.newaxis]
        residual=du_t +laplacian+u[0]**2+f_val # Computes the residual of the PDE.
        return residual 
    
    @partial(jit,static_argnames=["self"])
    def initial_constraint(self, x_t):
        '''
        Defines the initial constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the initial constraint.
        '''
        t = x_t[:, -1]
        x_1 = x_t[:, 0]
        x_2 = x_t[:, 1]
        result = -jnp.sin(jnp.pi*x_1/2)*jnp.cos(jnp.pi*x_2/2)*jnp.exp(-t)
        return result[:,jnp.newaxis]

    @partial(jit,static_argnames=["self"])
    def terminal_constraint(self, x_t):
        '''
        Defines the terminal constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the terminal constraint.
        '''
        t = x_t[:, -1]
        x_1 = x_t[:, 0]
        x_2 = x_t[:, 1]
        result = -jnp.sin(jnp.pi*x_1/2)*jnp.cos(jnp.pi*x_2/2)*jnp.exp(-t)
        return result[:,jnp.newaxis]
    
    @partial(jit,static_argnames=["self"])
    def Dirichlet_boundary_constraint(self, x_t):
        '''
        Defines the Dirichlet boundary constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the Dirichlet boundary constraint.
        '''
        t = x_t[:, -1]
        x_1 = x_t[:, 0]
        x_2 = x_t[:, 1]
        result = -jnp.sin(jnp.pi*x_1/2)*jnp.cos(jnp.pi*x_2/2)*jnp.exp(-t)
        return result[:,jnp.newaxis]

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
        return jnp.sqrt(2)
    
    @partial(jit,static_argnames=["self"])    
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
        x_1 = x_t[:, 0]
        x_2 = x_t[:, 1]
        t = x_t[:, -1]
        f_val = (jnp.pi**2-2)*jnp.sin(jnp.pi*x_1/2)*jnp.cos(jnp.pi*x_2/2)*jnp.exp(-t) - 4* (jnp.sin(jnp.pi*x_1/2)**2)*jnp.cos(jnp.pi*x_2/2)*jnp.exp(-2*t)
        return f_val[:,jnp.newaxis]+u**2
    
    @partial(jit,static_argnames=["self"])    
    def exact_solution(self, x_t):
        '''
        Computes the exact solution of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An arrary of shape (batch_size, n_ouput), representing the exact solution.
        '''
        t = x_t[:, -1]
        x_1 = x_t[:, 0]
        x_2 = x_t[:, 1]
        result = -jnp.sin(jnp.pi*x_1/2)*jnp.cos(jnp.pi*x_2/2)*jnp.exp(-t)
        return result[:,jnp.newaxis]
    
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
        timedomain = dde.geometry.TimeDomain(t0, T) # Defines the time domain.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain) # Combines spatial and time domains.
        self.geomx=spacedomain
        self.geomt=timedomain
        return geom
    
    def generate_data(self, num_domain=2500):
        '''
        Generates data for training the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        
        Returns:
        - data (dde.data.TimePDE): A TimePDE object containing the training data.
        '''
        geom=self.geometry() # Defines the geometry of the domain.
        # self.terminal_condition() # Generates terminal condition.
        self.Dirichlet_boundary_condition() # Generates Dirichlet boundary condition.
        self.initial_condition() # Generate initial condition
        # self.data_loss() # Generates data loss. 
        data = dde.data.TimePDE(
                                geom, # Geometry of the domain.
                                self.PDE_loss, # PDE loss function.
                                [self.ic, self.D_bc], # Additional conditions.
                                num_domain=num_domain, # Number of domain points.
                                num_boundary=100, # Number of boundary points.
                                num_initial=160,  # Number of initial points.
                                solution=self.exact_solution   # Incorporates exact solution for error metrics.
                            )
        return data
    
class Linear_Convection_Diffusion(Equation):
    '''
    Linear Convection Diffusion equation.
    '''
    def __init__(self, n_input, n_output=1):
        '''
        Initializes the PDE example with specified input and output dimensions.
        
        Parameters:
        - n_input (int): The dimension of the input space, including the time dimension.
        - n_output (int): The dimension of the output space. Defaults to 1.
        '''
        super().__init__(n_input, n_output)
        self.uncertainty = 0.5 * self.n_input
        self.norm_estimation = 0.5 * self.n_input
    
    def PDE_loss(self, x_t,u):
        '''
        Calculates the PDE loss for given inputs.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        
        Returns:
        - residual (tensor): The residual of the PDE of shape (batch_size, n_output).
        '''
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1)[0] # Computes the time derivative of u.
        laplacian=0
        div = 0
        d = self.n_input-1
        MC = int(d/4)
        # randomly choose MC dims to compute hessian and div
        idx_list = np.random.choice(self.n_input-1, MC, replace=False)
        for k in idx_list: # Accumulates laplacian and divergence over spatial dimensions.
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k)[0] # Computes the laplacian of z.
            div += dde.grad.jacobian(u, x_t, i=0, j=k)[0] # Computes the divergence of u.
        laplacian *= d/MC
        div *= d/MC
        residual=du_t +laplacian- (1/d)*div # Computes the residual of the PDE.
        return residual 
    
    @partial(jit,static_argnames=["self"])
    def initial_constraint(self, x_t):
        '''
        Defines the initial constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the initial constraint.
        '''
        x = x_t[:, :-1]
        result = jnp.sum(x, axis=1)
        return result[:,jnp.newaxis]

    @partial(jit,static_argnames=["self"])
    def terminal_constraint(self, x_t):
        '''
        Defines the terminal constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the terminal constraint.
        '''
        x = x_t[:, :-1]
        result = jnp.sum(x, axis=1) + self.T
        return result[:,jnp.newaxis]
    
    @partial(jit,static_argnames=["self"])
    def Dirichlet_boundary_constraint(self, x_t):
        '''
        Defines the Dirichlet boundary constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the Dirichlet boundary constraint.
        '''
        x = x_t[:, :-1]
        t = x_t[:, -1]
        result = jnp.sum(x, axis=1) + t
        return result[:,jnp.newaxis]

    def mu(self, x_t=0):
        '''
        Returns the drift coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The drift coefficient.
        '''
        d = self.n_input-1
        return -1/d
    
    def sigma(self, x_t=0):
        '''
        Returns the diffusion coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The diffusion coefficient.
        '''
        return jnp.sqrt(2)
    
    @partial(jit,static_argnames=["self"])    
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
        return jnp.zeros_like(u)
    
    @partial(jit,static_argnames=["self"])    
    def exact_solution(self, x_t):
        '''
        Computes the exact solution of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An arrary of shape (batch_size, n_ouput), representing the exact solution.
        '''
        x = x_t[:, :-1]
        t = x_t[:, -1]
        result = jnp.sum(x, axis=1) + t
        return result[:,jnp.newaxis]
    
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
        spacedomain = dde.geometry.Hypercube([0]*(self.n_input-1), [0.5]*(self.n_input-1)) # Defines the spatial domain, for train
        timedomain = dde.geometry.TimeDomain(t0, T) # Defines the time domain.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain) # Combines spatial and time domains.
        self.geomx=spacedomain
        self.geomt=timedomain
        return geom
    
    def generate_data(self, num_domain=50000):
        '''
        Generates data for training the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        
        Returns:
        - data (dde.data.TimePDE): A TimePDE object containing the training data.
        '''
        geom=self.geometry() # Defines the geometry of the domain.
        # self.terminal_condition() # Generates terminal condition.
        self.Dirichlet_boundary_condition() # Generates Dirichlet boundary condition.
        # self.initial_condition() # Generate initial condition
        # self.data_loss() # Generates data loss. 
        data = dde.data.TimePDE(
                                geom, # Geometry of the domain.
                                self.PDE_loss, # PDE loss function.
                                [self.D_bc], # Additional conditions.
                                num_domain=num_domain, # Number of domain points.
                                num_boundary=100, # Number of boundary points.
                                num_initial=0,  # Number of initial points.
                                solution=self.exact_solution   # Incorporates exact solution for error metrics.
                            )
        return data
    
class LQG(Equation):
    '''
    LQG equation.
    '''
    def __init__(self, n_input, n_output=1):
        '''
        Initializes the PDE example with specified input and output dimensions.
        
        Parameters:
        - n_input (int): The dimension of the input space, including the time dimension.
        - n_output (int): The dimension of the output space. Defaults to 1.
        '''
        super().__init__(n_input, n_output)
        self.uncertainty = 1e-1
        self.norm_estimation = 1
    
    def PDE_loss(self, x_t,u):
        '''
        Calculates the PDE loss for given inputs.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        
        Returns:
        - residual (tensor): The residual of the PDE of shape (batch_size, n_output).
        '''
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1)[0] # Computes the time derivative of u.
        laplacian=0
        d = self.n_input-1
        grad_norm_square = 0
        MC = int(self.n_input/4)
        # randomly choose MC dims to compute hessian and div
        idx_list = np.random.choice(self.n_input-1, MC, replace=False)
        for k in idx_list: # Accumulates laplacian and divergence over spatial dimensions.
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k)[0] # Computes the laplacian of z.
            grad_norm_square += dde.grad.jacobian(u, x_t, i=0, j=k)[0]**2
        laplacian *= d/MC
        grad_norm_square *= d/MC
        residual=du_t +laplacian- grad_norm_square # Computes the residual of the PDE.
        return residual 

    @partial(jit,static_argnames=["self"])
    def terminal_constraint(self, x_t):
        '''
        Defines the terminal constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the terminal constraint.
        '''
        # # HJB-Log
        # x = x_t[:, :-1]
        # result = jnp.log((1+jnp.linalg.norm(x,axis=1)**2)/2)
        # return result[:,jnp.newaxis]

        # HJB-Rosenbrock
        x = x_t[:, :-1]
        # g(\boldsymbol{x})=\log\left(\frac{1+\sum_{i=1}^{d-1}\left[c_{1,i}(\boldsymbol{x}_i-\boldsymbol{x}_{i+1})^2+c_{2,i}\boldsymbol{x}_{i+1}^2\right]}{2}\right)
        c1 = random.uniform(random.PRNGKey(0), (x.shape[0],x.shape[1]-1),minval=0.5,maxval=1.5)
        c2 = random.uniform(random.PRNGKey(1), (x.shape[0],x.shape[1]-1),minval=0.5,maxval=1.5)
        result = jnp.log((1+jnp.sum(c1*(x[:, :-1]-x[:, 1:])**2+c2*x[:, 1:]**2,axis=1))/2)
        return result[:,jnp.newaxis]

    @partial(jit,static_argnames=["self"])
    def Dirichlet_boundary_constraint(self, x_t):
        '''
        Defines the Dirichlet boundary constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the Dirichlet boundary constraint.
        '''
        sample_num = int(100 * (self.n_input-1)) 
        x = x_t[:, jnp.newaxis, :-1]
        t = x_t[:, jnp.newaxis, -1]
        simulated_x = x + jnp.sqrt(2)*random.normal(random.PRNGKey(0),(x.shape[0],sample_num,x.shape[1]))*jnp.sqrt(self.T-t[:,jnp.newaxis])
        inside = jnp.exp(-jit(vmap(self.terminal_constraint,in_axes=1,out_axes=1))(simulated_x))
        result = -jnp.log(jnp.mean(inside,axis=1))
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
        return jnp.sqrt(2)
    
    @partial(jit,static_argnames=["self"])    
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
        return -(jnp.linalg.norm(z,axis=1)**2)/2
    
    @partial(jit,static_argnames=["self"])    
    def exact_solution(self, x_t):
        '''
        Computes the exact solution of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An arrary of shape (batch_size, n_ouput), representing the exact solution.
        '''
        sample_num = int(100 * (self.n_input-1)) 
        x = x_t[:, jnp.newaxis, :-1]
        t = x_t[:, jnp.newaxis, -1]
        simulated_x = x + jnp.sqrt(2)*random.normal(random.PRNGKey(0),(x.shape[0],sample_num,x.shape[1]))*jnp.sqrt(self.T-t[:,jnp.newaxis])
        inside = jnp.exp(-jit(vmap(self.terminal_constraint,in_axes=1,out_axes=1))(simulated_x))
        result = -jnp.log(jnp.mean(inside,axis=1))
        return result
    
    def test_geometry(self, t0=0, T=1):
        '''
        Defines the geometry of the domain for the PDE.
        
        Parameters:
        - t0 (float): Initial time.
        - T (float): Terminal time.
        
        Returns:
        - geom (dde.geometry.GeometryXTime): A GeometryXTime object representing the domain.
        '''
        self.t0 = t0
        self.test_T = T
        self.test_radius = 1
        d = self.n_input -1
        spacedomain = dde.geometry.Hypersphere([0]*d,1)
        timedomain = dde.geometry.TimeDomain(t0, self.test_T)  # Defines the time domain for test.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain)  # Combines spatial and time domains.
        self.test_geomx = spacedomain
        self.test_geomt = timedomain
        return geom  

    def geometry(self,t0=0,T=1):
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
        d = self.n_input -1
        spacedomain = dde.geometry.Hypersphere([0]*d,1)
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
        # self.terminal_condition() # Generates terminal condition.
        self.Dirichlet_boundary_condition() # Generates Dirichlet boundary condition.
        # self.initial_condition() # Generate initial condition
        # self.data_loss() # Generates data loss. 
        data = dde.data.TimePDE(
                                geom, # Geometry of the domain.
                                self.PDE_loss, # PDE loss function.
                                [self.D_bc], # Additional conditions.
                                num_domain=num_domain, # Number of domain points.
                                num_boundary=1000, # Number of boundary points.
                                num_initial=0,  # Number of initial points.
                                solution=self.exact_solution   # Incorporates exact solution for error metrics.
                            )
        return data
    
class OScillating_Solution(Equation):
    '''
    Diffusion Reaction equation with ocilating solution.
    '''
    def __init__(self, n_input, n_output=1):
        '''
        Initializes the PDE example with specified input and output dimensions.
        
        Parameters:
        - n_input (int): The dimension of the input space, including the time dimension.
        - n_output (int): The dimension of the output space. Defaults to 1.
        '''
        super().__init__(n_input, n_output)
        self.uncertainty = 1e-2
        self.norm_estimation = 1
    
    def PDE_loss(self, x_t,u):
        '''
        Calculates the PDE loss for given inputs.
        
        Parameters:
        - x_t (tensor): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (tensor): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        
        Returns:
        - residual (tensor): The residual of the PDE of shape (batch_size, n_output).
        '''
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1)[0] # Computes the time derivative of u.
        laplacian=0
        d = self.n_input-1
        MC = int(self.n_input/4)
        # randomly choose MC dims to compute hessian and div
        idx_list = np.random.choice(self.n_input-1, MC, replace=False)
        for k in idx_list: # Accumulates laplacian and divergence over spatial dimensions.
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k)[0] # Computes the laplacian of z.
        laplacian *= d/MC
        residual=du_t +0.5* laplacian+ jnp.minimum(1,(u[0]-self.exact_solution(x_t))**2) # Computes the residual of the PDE.
        return residual 

    @partial(jit,static_argnames=["self"])
    def terminal_constraint(self, x_t):
        '''
        Defines the terminal constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the terminal constraint.
        '''
        kappa = 1.6
        lamb = 0.1 
        T = self.T
        d = self.n_input-1
        # u^\star(t,x)=\kappa+\sin\left(\lambda\sum_{i=1}^dx_i\right)\exp\left(\frac{\lambda^2d(t-T)}{2}\right)
        x = x_t[:, :-1]
        t = x_t[:, -1]
        result = kappa + jnp.sin(lamb*jnp.sum(x,axis=1))*jnp.exp(lamb**2*d*(t-T)/2)
        return result[:,jnp.newaxis]

    @partial(jit,static_argnames=["self"])
    def Dirichlet_boundary_constraint(self, x_t):
        '''
        Defines the Dirichlet boundary constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the Dirichlet boundary constraint.
        '''
        kappa = 1.6
        lamb = 0.1 
        T = self.T
        d = self.n_input-1
        # u^\star(t,x)=\kappa+\sin\left(\lambda\sum_{i=1}^dx_i\right)\exp\left(\frac{\lambda^2d(t-T)}{2}\right)
        x = x_t[:, :-1]
        t = x_t[:, -1]
        result = kappa + jnp.sin(lamb*jnp.sum(x,axis=1))*jnp.exp(lamb**2*d*(t-T)/2)
        return result[:,jnp.newaxis]

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
        return 1
    
    @partial(jit,static_argnames=["self"])    
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
        return jnp.minimum(1,(u-self.exact_solution(x_t))**2)
    
    @partial(jit,static_argnames=["self"])    
    def exact_solution(self, x_t):
        '''
        Computes the exact solution of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An arrary of shape (batch_size, n_ouput), representing the exact solution.
        '''
        kappa = 1.6
        lamb = 0.1 
        T = self.T
        d = self.n_input-1
        # u^\star(t,x)=\kappa+\sin\left(\lambda\sum_{i=1}^dx_i\right)\exp\left(\frac{\lambda^2d(t-T)}{2}\right)
        x = x_t[:, :-1]
        t = x_t[:, -1]
        result = kappa + jnp.sin(lamb*jnp.sum(x,axis=1))*jnp.exp(lamb**2*d*(t-T)/2)
        return result[:,jnp.newaxis]
    
    def test_geometry(self, t0=0, T=1):
        '''
        Defines the geometry of the domain for the PDE.
        
        Parameters:
        - t0 (float): Initial time.
        - T (float): Terminal time.
        
        Returns:
        - geom (dde.geometry.GeometryXTime): A GeometryXTime object representing the domain.
        '''
        self.t0 = t0
        self.test_T = T
        self.test_radius = 1
        d = self.n_input -1
        spacedomain = dde.geometry.Hypersphere([0]*d,1)
        timedomain = dde.geometry.TimeDomain(t0, self.test_T)  # Defines the time domain for test.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain)  # Combines spatial and time domains.
        self.test_geomx = spacedomain
        self.test_geomt = timedomain
        return geom  

    def geometry(self,t0=0,T=1):
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
        d = self.n_input -1
        spacedomain = dde.geometry.Hypersphere([0]*d,1)
        timedomain = dde.geometry.TimeDomain(t0, T) # Defines the time domain.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain) # Combines spatial and time domains.
        self.geomx=spacedomain
        self.geomt=timedomain
        return geom
    
    def generate_data(self, num_domain=1000):
        '''
        Generates data for training the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        
        Returns:
        - data (dde.data.TimePDE): A TimePDE object containing the training data.
        '''
        geom=self.geometry() # Defines the geometry of the domain.
        # self.terminal_condition() # Generates terminal condition.
        self.Dirichlet_boundary_condition() # Generates Dirichlet boundary condition.
        # self.initial_condition() # Generate initial condition
        # self.data_loss() # Generates data loss. 
        data = dde.data.TimePDE(
                                geom, # Geometry of the domain.
                                self.PDE_loss, # PDE loss function.
                                [self.D_bc], # Additional conditions.
                                num_domain=num_domain, # Number of domain points.
                                num_boundary=1000, # Number of boundary points.
                                num_initial=0,  # Number of initial points.
                                solution=self.exact_solution   # Incorporates exact solution for error metrics.
                            )
        return data