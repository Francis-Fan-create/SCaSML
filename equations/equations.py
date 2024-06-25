import deepxde as dde
import numpy as np
import torch


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
        PINN loss in the PDE, used in ScaML to calculate epsilon.
        
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

    def boundary_constraint(self, x_t):
        """
        Boundary constraint in the PDE.
        
        Args:
            x_t (ndarray): The input data at the boundary, shape (n_samples, n_input).
            
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

    def boundary_condition(self):
        """
        Boundary condition of the PDE, using soft constraint.
        
        Returns:
            dde.icbc.DirichletBC: The boundary condition object.
            
        Raises:
            NotImplementedError: If the boundary_constraint or geometry method is not implemented.
        """
        if hasattr(self, 'boundary_constraint') and hasattr(self, 'geometry'):
            bc = dde.icbc.DirichletBC(self.geometry(), self.boundary_constraint, lambda _, on_boundary: on_boundary)  # need to be enforced on generate_data method
            self.bc = bc
            return bc
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
        spacedomain = dde.geometry.Hypercube([-0.5]*(self.n_input-1), [0.5]*(self.n_input-1)) # Defines the spatial domain.
        timedomain = dde.geometry.TimeDomain(t0, T) # Defines the time domain.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain) # Combines spatial and time domains.
        self.geomx=spacedomain
        self.geomt=timedomain
        return geom
    
    def generate_data(self, num_domain=2000):
        '''
        Generates data for training the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        
        Returns:
        - data (dde.data.TimePDE): A TimePDE object containing the training data.
        '''
        geom=self.geometry() # Defines the geometry of the domain.
        self.terminal_condition() # Generates terminal condition.
        self.boundary_condition()  # Generates boundary condition.
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
    
class Explicit_Solution_Example_Rescale(Equation):
    '''
    
    Example of high-dimensional PDE with exact solution in rescaled version.

    '''
    def __init__(self, n_input, n_output=1):
        '''Initializes the equation with input and output dimensions.'''
        super().__init__(n_input, n_output)

    def PDE_loss(self, x_t, u, z):
        '''Calculates the PDE loss.
        
        Args:
            x_t (tensor): Input tensor of shape (batch_size, n_input).
            u (tensor): Solution tensor of shape (batch_size, n_output).
            z (tensor): Gradient tensor of shape (batch_size, n_input-1).
        
        Returns:
            residual(tensor): Residual tensor of shape (batch_size,n_output).
        '''
        # Compute the time derivative of u with respect to the last dimension of x_t
        du_t = dde.grad.jacobian(u, x_t, i=0, j=self.n_input-1)
        laplacian = 0
        div = 0
        dim = self.n_input-1
        for k in range(self.n_input-1):  # Accumulate laplacian and divergence using autograd
            laplacian += dde.grad.jacobian(z, x_t, i=k, j=k)
            div += dde.grad.jacobian(u, x_t, i=0, j=k)
        # Calculate the residual of the PDE
        residual = du_t + (self.sigma()**2 * dim * u - 1 - dim * self.sigma()**2 / 2) * div + (dim * self.sigma())**2 / 2 * laplacian
        return residual 

    def gPDE_loss(self, x_t, u):
        '''Calculates the gPDE loss using gPINN.
        
        Args:
            x_t (tensor): Input tensor of shape (batch_size, n_input).
            u (tensor): Solution tensor of shape (batch_size, n_output).
        
        Returns:
            g_loss(list of ndarray): List of gradient loss tensors, each of shape (batch_size, n_output).
        '''
        # Compute the time derivative of u with respect to the last dimension of x_t
        du_t = dde.grad.jacobian(u, x_t, i=0, j=self.n_input-1)
        laplacian = 0
        div = 0
        dim = self.n_input-1
        for k in range(self.n_input-1):  # Accumulate laplacian and divergence using autograd
            laplacian += dde.grad.hessian(u, x_t, i=k, j=k)
            div += dde.grad.jacobian(u, x_t, i=0, j=k)
        # Calculate the residual of the PDE
        residual = du_t + (self.sigma()**2 * dim * u - 1 - dim * self.sigma()**2 / 2) * div + (dim * self.sigma())**2 / 2 * laplacian
        g_loss = []
        for k in range(self.n_input-1):  # Compute gradient loss for each dimension
            g_loss.append(0.01 * dde.grad.jacobian(residual, x_t, i=0, j=k))
        g_loss.append(residual)
        return g_loss

    def terminal_constraint(self, x_t):
        '''Defines the terminal constraint for the PDE.
        
        Args:
            x_t (ndarray): Input tensor of shape (batch_size, n_input).
        
        Returns:
            ndarray: Constraint tensor of shape (batch_size,).
        '''
        dim = self.n_input-1
        # Calculate the terminal constraint
        result = 1 - 1 / (1 + np.exp(x_t[:, -1] + (1 / dim) * np.sum(x_t[:, :self.n_input-1], axis=1)))
        return result 

    def mu(self, x_t=0):
        '''Returns the drift term, mu.
        
        Args:
            x_t (optional): Not used.
        
        Returns:
            float: The drift term.
        '''
        return 0

    def sigma(self, x_t=0):
        '''Returns the volatility term, sigma.
        
        Args:
            x_t (optional): Not used.
        
        Returns:
            float: The volatility term.
        '''
        return 0.25

    def f(self, x_t, u, z):
        '''Defines the generator term for the PDE.
        
        Args:
            x_t (ndarray): Input tensor of shape (batch_size, n_input).
            u (ndarray): Solution tensor of shape (batch_size, n_output).
            z (ndarray): Gradient tensor of shape (batch_size, n_input-1).
        
        Returns:
            ndarray: Generator term tensor of shape (batch_size, n_output).
        '''
        dim = self.n_input-1
        result=(1/dim)*self.sigma() * (u - 0.5-(1/self.sigma())) * (np.sum(z, axis=1,keepdims=True))
        return result
    
    def exact_solution(self, x_t):
        '''Calculates the exact solution of the example.
        
        Args:
            x_t (ndarray): Input tensor of shape (batch_size, n_input).
        
        Returns:
            ndarray: Exact solution tensor of shape (batch_size, n_output).
        '''
        dim = self.n_input-1
        t = x_t[:, -1]
        x = x_t[:, :-1]
        sum_x = np.sum(x, axis=1)
        exp_term = np.exp(t + (1 / dim) * sum_x)
        result = 1 - 1 / (1 + exp_term)
        return result
    
    def geometry(self, t0=0, T=0.5):
        '''Defines the geometry of the domain, which is a hypercube.
        
        Args:
            t0 (float): Initial time.
            T (float): Terminal time.
        
        Returns:
            dde.geometry.GeometryXTime: Combined spatial and temporal domain.
        '''
        self.t0 = t0
        self.T = T
        spacedomain = dde.geometry.Hypercube([-0.5] * (self.n_input-1), [0.5] * (self.n_input-1))
        timedomain = dde.geometry.TimeDomain(t0, T)
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain)  # Combine both domains
        self.geomx = spacedomain
        self.geomt = timedomain
        return geom
    
    def generate_data(self, num_domain=2000):
        '''Generates data for training the model.
        
        Args:
            num_domain (int): Number of points to sample in the domain.
        
        Returns:
            dde.data.TimePDE: Data object for time-dependent PDE.
        '''
        geom = self.geometry()
        self.terminal_condition()  # Generate terminal condition
        self.boundary_condition()  # Generate boundary condition
        data = dde.data.TimePDE(
            geom,  # Geometry of the boundary condition and terminal condition
            self.gPDE_loss,  # g_pde residual
            [self.tc],  # Additional conditions other than PDE loss
            num_domain=num_domain,  # Sample how many points in the domain
            num_boundary=0,  # Sample how many points on the boundary
            num_initial=0,  # Sample how many points for the initial time
            anchors=self.my_data,  # Enforce terminal points
            solution=self.exact_solution,  # Incorporate authentic solution to evaluate error metrics
            num_test=None  # Sample how many points for testing. If None, then the training point will be used.
        )
        return data
    
class LQG(Equation):
    '''Linear Quadratic Gaussian Control Problem
    
    Attributes:
        n_input (int): Number of input dimensions.
        n_output (int): Number of output dimensions, default is 1.
    '''
    def __init__(self, n_input, n_output=1):
        '''Initialize the LQG problem with given input and output dimensions.'''
        super().__init__(n_input, n_output)
        
    def PDE_loss(self, x_t, u, z):
        '''Compute the PDE loss for the LQG problem.
        
        Args:
            x_t (tensor): The input tensor of shape (batch_size, n_input).
            u (tensor): The control tensor of shape (batch_size, n_output).
            z (tensor): The state tensor of shape (batch_size, n_input-1).
            
        Returns:
            ndarray: The residual tensor of shape (batch_size, n_output).
        '''
        # Compute the gradient of u with respect to the last input dimension
        du_t = dde.grad.jacobian(u, x_t, i=0, j=self.n_input-1)
        laplacian = 0
        grad_norm = 0
        # Compute the Laplacian and gradient norm using autograd
        for k in range(self.n_input-1):
            laplacian += dde.grad.jacobian(z, x_t, i=k, j=k)
            grad_norm += dde.grad.jacobian(u, x_t, i=0, j=k)**2
        residual = du_t + laplacian - grad_norm
        return residual
    
    def gPDE_loss(self, x_t, u):
        '''Compute the gPDE loss for the LQG problem.
        
        Args:
            x_t (tensor): The input tensor of shape (batch_size, n_input).
            u (tensor): The control tensor of shape (batch_size, n_output).
            
        Returns:
            list[tensor]: A list of gradient tensors, each of shape (batch_size, n_output).
        '''
        # Compute the gradient and Laplacian for gPDE loss
        du_t = dde.grad.jacobian(u, x_t, i=0, j=self.n_input-1)
        laplacian = 0
        grad_norm = 0
        for k in range(self.n_input-1):
            laplacian += dde.grad.hessian(u, x_t, i=k, j=k)
            grad_norm += dde.grad.jacobian(u, x_t, i=0, j=k)**2
        residual = du_t + laplacian - grad_norm
        g_loss = []
        for k in range(self.n_input-1):
            g_loss.append(0.01 * dde.grad.jacobian(residual, x_t, i=0, j=k))
        g_loss.append(residual)
        return g_loss
    
    def terminal_constraint(self, x_t):
        '''Compute the terminal constraint for the LQG problem.
        
        Args:
            x_t (ndarray): The input tensor of shape (batch_size, n_input).
            
        Returns:
            ndarray: The result tensor of shape (batch_size,).
        '''
        # Compute the terminal constraint using the norm of x_t
        result = np.log((1 + np.linalg.norm(x_t[:, :self.n_input-1], axis=1)**2) / 2)
        return result
    
    def mu(self, x_t=0):
        '''Compute the mean of the distribution.
        
        Args:
            x_t (int, optional): Dummy input. Defaults to 0.
            
        Returns:
            int: The mean value.
        '''
        return 0
    
    def sigma(self, x_t=0):
        '''Compute the standard deviation of the distribution.
        
        Args:
            x_t (int, optional): Dummy input. Defaults to 0.
            
        Returns:
            float: The standard deviation value.
        '''
        return np.sqrt(2)
    
    def f(self, x_t, u, z):
        '''Compute the generator term for the PDE.
        
        Args:
            x_t (ndarray): The input tensor of shape (batch_size, n_input).
            u (ndarray): The solution tensor, unused in this method.
            z (ndarray): The grad tensor of shape (batch_size, n_input-1).
            
        Returns:
            ndarray: The result tensor of shape (batch_size, 1).
        '''
        # Compute the generator term using the norm of z
        result = -np.linalg.norm(z, axis=1, keepdims=True)**2
        return result
    
    def exact_solution(self, x_t):
        '''Compute the exact solution for the LQG problem.
        
        Args:
            x_t (ndarray): The input tensor of shape (batch_size, n_input).
            
        Returns:
            ndarray: The solution tensor of shape (batch_size, n_output).
        '''
        # Compute the exact solution using Monte Carlo simulation
        x = x_t[:, :self.n_input-1]
        t = x_t[:, -1]
        scale = np.sqrt(self.T - t)
        MC = 300
        X = np.repeat(x.reshape(x.shape[0], x.shape[1], 1), MC, axis=2)
        W = np.random.normal(0, 1, X.shape)
        disturbed_X = X + np.sqrt(2) * scale[:, np.newaxis, np.newaxis] * W
        exp_value = np.exp(-self.g(disturbed_X))
        result = -np.log(np.mean(exp_value, axis=1))
        return result[:, np.newaxis]
    
    def geometry(self, t0=0, T=0.5):
        '''Define the geometry of the domain for the LQG problem.
        
        Args:
            t0 (float): The initial time.
            T (float): The terminal time.
            
        Returns:
            GeometryXTime: The combined spatial and temporal domain.
        '''
        # Define the spatial and temporal domains
        self.t0 = t0
        self.T = T
        spacedomain = dde.geometry.Hypercube([-0.5] * (self.n_input-1), [0.5] * (self.n_input-1))
        timedomain = dde.geometry.TimeDomain(t0, T)
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain)
        self.geomx = spacedomain
        self.geomt = timedomain
        return geom
    
    def generate_data(self, num_domain=2000):
        '''Generate data for training the LQG model.
        
        Args:
            num_domain (int): Number of points to sample in the domain.
            
        Returns:
            TimePDE: The data object containing training data for the LQG problem.
        '''
        geom = self.geometry()
        # Generate terminal and boundary conditions
        self.terminal_condition()
        self.boundary_condition()
        data = dde.data.TimePDE(
            geom,
            self.gPDE_loss,
            [self.tc],
            num_domain=num_domain,
            num_boundary=0,
            num_initial=0,
            anchors=self.my_data,
            solution=self.exact_solution,
            num_test=None
        )
        return data
    
class Complicated_HJB(Equation):
    '''Complicated HJB equation with exact solution. This class defines a complex Hamilton-Jacobi-Bellman (HJB) equation and provides methods to compute losses, exact solutions, and generate data for training and testing purposes.
    
    Attributes:
        n_input (int): Number of input dimensions.
        n_output (int): Number of output dimensions, default is 1.
    '''
    def __init__(self, n_input, n_output=1):
        '''Initializes the Complicated_HJB class with input and output dimensions.
        
        Args:
            n_input (int): Number of input dimensions.
            n_output (int): Number of output dimensions, default is 1.
        '''
        super().__init__(n_input, n_output)
        
    def PDE_loss(self, x_t,u,z):
        '''Computes the PDE loss for the given inputs.
        
        Args:
            x_t (Tensor): The input tensor with dimensions [batch_size, n_input].
            u (Tensor): The solution tensor with dimensions [batch_size, n_output].
            z (Tensor): The gradient tensor with dimensions [batch_size, n_input-1].
        
        Returns:
            Tensor: The residual tensor with dimensions [batch_size, n_output].
        '''
        # Compute the time derivative of u with respect to the last input dimension
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1)
        laplacian=0
        grad_norm=0
        dim=self.n_input-1
        for k in range(self.n_input-1): # Accumulate laplacian and gradient norm using autograd
            laplacian +=dde.grad.jacobian(z, x_t, i=k, j=k)
            grad_norm +=torch.abs(dde.grad.jacobian(u, x_t, i=0, j=k)) 
        residual=du_t + laplacian-(1/dim)*grad_norm +2
        return residual
    
    def gPDE_loss(self, x_t,u):
        '''Computes the generalized PDE (gPDE) loss for the given inputs.
        
        Args:
            x_t (Tensor): The input tensor with dimensions [batch_size, n_input].
            u (Tensor): The solution tensor with dimensions [batch_size, n_output].
        
        Returns:
            List[Tensor]: A list of tensors representing the gPDE loss for each input dimension and the residual.
        '''
        # Compute the time derivative of u with respect to the last input dimension
        du_t = dde.grad.jacobian(u,x_t,i=0,j=self.n_input-1)
        laplacian=0
        grad_norm=0
        dim=self.n_input-1
        for k in range(self.n_input-1): # Accumulate laplacian and gradient norm using autograd
            laplacian +=dde.grad.hessian(u, x_t, i=k, j=k)
            grad_norm +=torch.abs(dde.grad.jacobian(u, x_t, i=0, j=k)) 
        residual=du_t + laplacian-(1/dim)*grad_norm +2
        g_loss=[]
        for k in range(self.n_input-1):
            g_loss.append(0.01*dde.grad.jacobian(residual,x_t,i=0,j=k))
        g_loss.append(residual)
        return g_loss
    
    def terminal_constraint(self, x_t):
        '''Computes the terminal constraint for the given inputs.
        
        Args:
            x_t (ndarray): The input array with dimensions [batch_size, n_input].
        
        Returns:
            ndarray: The result array with dimensions [batch_size].
        '''
        # Sum over the input dimensions except the last one
        result= np.sum(x_t[:,:self.n_input-1],axis=1)
        return result
    
    def mu(self, x_t=0):
        '''Returns the mean of the process.
        
        Args:
            x_t (int, optional): Dummy input. Defaults to 0.
        
        Returns:
            int: The mean value, which is 0.
        '''
        return 0
    
    def sigma(self, x_t=0):
        '''Returns the standard deviation of the process.
        
        Args:
            x_t (int, optional): Dummy input. Defaults to 0.
        
        Returns:
            float: The standard deviation, which is sqrt(2).
        '''
        return np.sqrt(2)
    
    def f(self, x_t,u,z):
        '''Computes the generator term for the PDE.
        
        Args:
            x_t (ndarray): The input array with dimensions [batch_size, n_input].
            u (ndarray): Dummy input for compatibility.
            z (ndarray): The gradient array with dimensions [batch_size, n_input-1].
        
        Returns:
            ndarray: The result array with dimensions [batch_size, 1].
        '''
        dim=self.n_input-1
        result=-(1/dim)*np.sum(np.abs(z),axis=1) +2
        return result[:,np.newaxis]
    
    def exact_solution(self, x_t):
        '''Computes the exact solution for the given inputs.
        
        Args:
            x_t (ndarray): The input array with dimensions [batch_size, n_input].
        
        Returns:
            ndarray: The exact solution array with dimensions [batch_size, 1].
        '''
        x=x_t[:,:self.n_input-1]
        t=x_t[:,-1]
        result=np.sum(x,axis=1)+(self.T-t)
        return result[:,np.newaxis]
    
    def geometry(self,t0=0,T=0.5):
        '''Defines the geometry of the domain.
        
        Args:
            t0 (float, optional): The start time of the domain. Defaults to 0.
            T (float, optional): The end time of the domain. Defaults to 0.5.
        
        Returns:
            GeometryXTime: The combined spatial and temporal domain.
        '''
        self.t0=t0
        self.T=T
        spacedomain = dde.geometry.Hypercube([-0.5]*(self.n_input-1), [0.5]*(self.n_input-1)) 
        timedomain = dde.geometry.TimeDomain(t0, T) 
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain)
        self.geomx=spacedomain
        self.geomt=timedomain
        return geom
    
    def generate_data(self, num_domain=100):
        '''Generates data for training and testing.
        
        Args:
            num_domain (int, optional): Number of points to sample in the domain. Defaults to 100.
        
        Returns:
            TimePDE: The data object containing the generated data.
        '''
        geom=self.geometry()
        self.terminal_condition() # Generate terminal condition
        self.boundary_condition() # Generate boundary condition
        data = dde.data.TimePDE(
                                geom,
                                self.gPDE_loss,
                                [self.tc],
                                num_domain=num_domain,
                                num_boundary=0,
                                num_initial=0,
                                anchors=self.my_data,
                                solution=self.exact_solution,
                                num_test=None
                            )
        return data
    