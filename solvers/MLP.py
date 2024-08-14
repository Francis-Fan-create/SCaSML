import torch
import numpy as np
import deepxde as dde
import torch.nn as nn
from scipy.special import lambertw
from utils.log_variables import log_variables
class MLP(object):
    '''Multilevel Picard Iteration for high dimensional semilinear PDE'''
    #all the vectors uses rows as index and columns as dimensions
    def __init__(self, equation):
        '''
        Initialize the MLP parameters based on the given equation.
        
        Args:
            equation: An object containing the parameters and functions defining the equation to be solved by the MLP.
        '''
        # Initialize the MLP parameters from the equation object
        self.equation = equation
        self.sigma = equation.sigma  # Volatility parameter from the equation
        self.mu = equation.mu  # Drift parameter from the equation
        # self.f = equation.f  # Source term function of the equation
        # self.g = equation.g  # Terminal condition function of the equation
        equation.geometry()  # Initialize the geometry related parameters in the equation
        self.T = equation.T  # Terminal time
        self.t0 = equation.t0  # Initial time
        self.n_input = equation.n_input  # Number of input features
        self.n_output = equation.n_output  # Number of output features
        self.evaluation_counter=0 # Number of evaluations

    def f(self, x_t, u_breve, z_breve):
        '''
        Generator function of ScaSML, representing the light and large version.
        
        Parameters:
            x_t (ndarray): Input data of shape (batch_size, n_input).
            u_breve (ndarray): approximated u_ture-u_hat.
            z_breve (ndarray): approximated gradient of u_breve.
        
        Returns:
            ndarray: The output of the generator function of shape (batch_size,).
        '''
        batch_size=x_t.shape[0]
        self.evaluation_counter+=batch_size
        eq = self.equation
        return eq.f(x_t, u_breve, z_breve)
    
    def g(self, x_t):
        '''
        Terminal constraint function of ScaSML.
        
        Parameters:
            x_t (ndarray): Input data of shape (batch_size, n_input).
        
        Returns:
            ndarray: The output of the terminal constraint function of shape (batch_size,).
        '''
        batch_size=x_t.shape[0]
        self.evaluation_counter+=batch_size
        eq = self.equation
        return eq.g(x_t)
    
    def inverse_gamma(self, gamma_input):
        '''
        Compute the inverse of the gamma function for the given input.
        
        Args:
            gamma_input (np.ndarray): Input values for which to compute the inverse gamma function, shape (n,).
            
        Returns:
            np.ndarray: The computed inverse gamma values, shape (n,).
        '''
        c = 0.036534 # Constant to avoid singularity
        L = np.log((gamma_input + c) / np.sqrt(2 * np.pi))  # Logarithm part of the inverse gamma function
        return np.real(L / lambertw(L / np.e) + 0.5) # Compute and return the real part of the inverse gamma function
    
    def lgwt(self, N, a, b):
        '''
        Computes the Legendre-Gauss nodes and weights for numerical integration.
        
        Args:
            N (int): The number of nodes and weights to compute.
            a (float): The lower bound of the integration interval.
            b (float): The upper bound of the integration interval.
            
        Returns:
            tuple: A tuple containing two numpy arrays. The first array contains the nodes (shape: (N,)),
                and the second array contains the weights (shape: (N,)).
        '''
        N -= 1  # Adjust N for zero-based indexing
        N1, N2 = N + 1, N + 2  # Adjusted counts for nodes and weights
        xu = np.linspace(-1, 1, N1).reshape(1, -1)  # Initial uniform nodes on [-1, 1], reshaped to row vector
        # Initial guess for nodes using cosine transformation and correction term
        y = np.cos((2 * np.arange(0, N + 1, 1) + 1) * np.pi / (2 * N + 2)) + (0.27 / N1) * np.sin(np.pi * xu * N / N2)
        L = np.zeros((N1, N2))  # Initialize Legendre-Gauss Vandermonde Matrix
        Lp = np.zeros((N1, N2))  # Initialize derivative of LG Vandermonde Matrix
        y0 = 2  # Initial value for iteration comparison
        # Iterative computation using Newton-Raphson method
        while np.max(np.abs(y - y0)) > 2.2204e-16:
            L[:, 0] = 1
            Lp[:, 0] = 0
            L[:, 1] = y
            Lp[:, 1] = 1
            for k in range(2, N1 + 1):
                L[:, k] = ((2 * k - 1) * y * L[:, k - 1] - (k - 1) * L[:, k - 2]) / k
            Lp = (N2) * (L[:, N1 - 1] - y * L[:, N2 - 1]) / (1 - y ** 2)
            y0 = y
            y = y0 - L[:, N2 - 1] / Lp
        x = (a * (1 - y) + b * (1 + y)) / 2  # Map nodes to [a, b]
        w = (b - a) / ((1 - y ** 2) * (Lp ** 2)) * (N2 ** 2) / (N1 ** 2)  # Compute weights
        return x[0], w[0]

    def approx_parameters(self, rhomax):
        '''
        Approximates parameters for the multilevel Picard iteration.
        
        Args:
            rhomax (int): Maximum level of refinement.
            
        Returns:
            tuple: A tuple containing matrices for forward Euler steps (Mf), backward Euler steps (Mg),
                number of quadrature points (Q), quadrature points (c), and quadrature weights (w).
                Shapes are as follows: Mf, Mg, Q are (rhomax, rhomax), c and w are (qmax, qmax),
                where qmax is the maximum number of quadrature points across all levels.
        '''
        levels = list(range(1, rhomax + 1))  # Level list
        Q = np.zeros((rhomax, rhomax))  # Initialize matrix for number of quadrature points
        Mf = np.zeros((rhomax, rhomax))  # Initialize matrix for forward Euler steps
        Mg = np.zeros((rhomax, rhomax + 1))  # Initialize matrix for backward Euler steps
        for rho in range(1, rhomax + 1):
            for k in range(1, levels[rho - 1] + 1):
                Q[rho - 1][k - 1] = round(self.inverse_gamma(rho ** (k / 2)))  # Compute quadrature points
                Mf[rho - 1][k - 1] = round(rho ** (k / 2))  # Compute forward Euler steps
                Mg[rho - 1][k - 1] = round(rho ** (k - 1))  # Compute backward Euler steps
            Mg[rho - 1][rho] = rho ** rho  # Special case for backward Euler steps
        qmax = int(np.max(Q))  # Determine maximum number of quadrature points
        c = np.zeros((qmax, qmax))  # Initialize array for quadrature points
        w = np.zeros((qmax, qmax))  # Initialize array for quadrature weights
        for k in range(1, qmax + 1):
            ctemp, wtemp = self.lgwt(k, 0, self.T)  # Compute Legendre-Gauss nodes and weights
            c[:, k - 1] = np.concatenate([ctemp[::-1], np.zeros(qmax - k)])  # Store quadrature points
            w[:, k - 1] = np.concatenate([wtemp[::-1], np.zeros(qmax - k)])  # Store quadrature weights
        return Mf, Mg, Q, c, w

    def set_approx_parameters(self, rhomax):
        '''
        Sets the approximation parameters for the multilevel Picard iteration.
        This method should be called before solving the PDE.
        
        Args:
            rhomax (int): Maximum level of refinement.
        '''
        self.Mf, self.Mg, self.Q, self.c, self.w = self.approx_parameters(rhomax)  # Set approximation parameters
    
    # @log_variables
    def uz_solve(self, n, rho, x_t):
        '''
        Approximate the solution of the PDE, return the value of u(x_t) and z(x_t), batchwisely.
        
        Parameters:
            n (int): The index of summands in quadratic sum.
            rho (int): Current level.
            x_t (ndarray): A batch of spatial-temporal coordinates, shape (batch_size, n_input), where
                           batch_size is the number of samples in the batch and n_input is the number of input features (spatial dimensions + 1 for time).
        
        Returns:
            ndarray: The concatenated u and z values for each sample in the batch, shape (batch_size, 1+n_input-1).
                     Here, u is the approximate solution of the PDE at the given coordinates, and z is the associated spatial gradient.
        '''
        # Extract model parameters and functions
        Mf, Mg, Q, c, w = self.Mf, self.Mg, self.Q, self.c, self.w
        T = self.T  # Terminal time
        dim = self.n_input - 1  # Spatial dimensions
        batch_size = x_t.shape[0]  # Batch size
        sigma = self.sigma(x_t)  # Volatility, shape (batch_size, dim)
        mu= self.mu(x_t)  # Drift, shape (batch_size, dim)
        x = x_t[:, :-1]  # Spatial coordinates, shape (batch_size, dim)
        t = x_t[:, -1]  # Temporal coordinates, shape (batch_size,)
        f = self.f  # Generator term function
        g = self.g  # Terminal constraint function
        
        # Compute local time and weights
        cloc = (T - t)[:, np.newaxis, np.newaxis] * c[np.newaxis, :] / T + t[:, np.newaxis, np.newaxis]  # Local time, shape (batch_size, 1, 1)
        wloc = (T - t)[:, np.newaxis, np.newaxis] * w[np.newaxis, :] / T  # Local weights, shape (batch_size, 1)
        
        # Determine the number of Monte Carlo samples for backward Euler
        MC = int(Mg[rho - 1, n])  # Number of Monte Carlo samples, scalar
        
        # Generate Monte Carlo samples for backward Euler
        W = np.sqrt(T - t)[:, np.newaxis, np.newaxis] * np.random.normal(size=(batch_size, MC, dim))  # Brownian increments, shape (batch_size, MC, dim)
        X = np.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC, axis=1)  # Replicated spatial coordinates, shape (batch_size, MC, dim)
        disturbed_X = X + mu*(T-t)[:, np.newaxis, np.newaxis]+ sigma * W  # Disturbed spatial coordinates, shape (batch_size, MC, dim)
        
        # Initialize arrays for terminal and difference values
        terminals = np.zeros((batch_size, MC, 1))  # Terminal values, shape (batch_size, MC, 1)
        differences = np.zeros((batch_size, MC, 1))  # Differences, shape (batch_size, MC, 1)
        
        # Compute terminal and difference values for each Monte Carlo sample
        for i in range(MC):
            input_terminal = np.concatenate((X[:, i, :], np.full((batch_size, 1), T)), axis=1)  # Terminal spatial-temporal coordinates, shape (batch_size, n_input)
            disturbed_input_terminal = np.concatenate((disturbed_X[:, i, :], np.full((batch_size, 1), T)), axis=1)  # Disturbed terminal spatial-temporal coordinates, shape (batch_size, n_input)
            terminals[:, i, :] = g(input_terminal)[:, np.newaxis]  # Apply terminal constraint function, shape (batch_size, 1)
            differences[:, i, :] = (g(disturbed_input_terminal) - g(input_terminal))[:, np.newaxis]  # Compute differences, shape (batch_size, 1)
        
        # Compute u and z values
        u = np.mean(differences + terminals, axis=1)  # Mean over Monte Carlo samples, shape (batch_size, 1)
        # if (T - t).any() == 0:
        #     delta_t = (T - t + 1e-6)[:, np.newaxis]  # Avoid division by zero, shape (batch_size, 1)
        #     z = np.sum(differences * W, axis=1) / (MC * delta_t)  # Compute z values, shape (batch_size, dim)
        # else:
        #     z = np.sum(differences * W, axis=1) / (MC * (T - t)[:, np.newaxis])  # Compute z values, shape (batch_size, dim)
        delta_t = (T - t + 1e-6)[:, np.newaxis]  # Avoid division by zero, shape (batch_size, 1)
        z = np.sum(differences * W, axis=1) / (MC * delta_t)  # Compute z values, shape (batch_size, dim)        
        # Recursive call for n > 0
        if n <= 0:
            return np.concatenate((u, z), axis=-1)  # Concatenate u and z values, shape (batch_size, dim + 1)
        
        # Recursive computation for n > 0
        for l in range(n):
            q = int(Q[rho - 1, n - l - 1])  # Number of quadrature points, scalar
            d = cloc[:, :q, q - 1] - np.concatenate((t[:, np.newaxis], cloc[:, :q - 1, q - 1]), axis=1)  # Time steps, shape (batch_size, q)
            MC = int(Mf[rho - 1, n - l - 1])  # Number of Monte Carlo samples, scalar
            X = np.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC, axis=1)  # Replicated spatial coordinates, shape (batch_size, MC, dim)
            W = np.zeros((batch_size, MC, dim))  # Initialize Brownian increments, shape (batch_size, MC, dim)
            simulated = np.zeros((batch_size, MC, dim + 1))  # Initialize array for simulated values, shape (batch_size, MC, dim + 1)
            
            # Compute simulated values for each quadrature point
            for k in range(q):
                dW = np.sqrt(d[:, k])[:, np.newaxis, np.newaxis] * np.random.normal(size=(batch_size, MC, dim))  # Brownian increments for current time step, shape (batch_size, MC, dim)
                W += dW  # Accumulate Brownian increments
                X += mu*(d[:, k])[:,np.newaxis,np.newaxis]+sigma * dW  # Update spatial coordinates
                co_solver_l = lambda X_t: self.uz_solve(n=l, rho=rho, x_t=X_t)  # Co-solver for level l
                co_solver_l_minus_1 = lambda X_t: self.uz_solve(n=l - 1, rho=rho, x_t=X_t)  # Co-solver for level l - 1
                input_intermediates=np.zeros((batch_size,MC,dim+1))
                # Compute u and z values for current quadrature point
                for i in range(MC):
                    input_intermediate = np.concatenate((X[:, i, :], cloc[:, k, q - 1][:, np.newaxis]), axis=1)  # Intermediate spatial-temporal coordinates, shape (batch_size, n_input)
                    simulated[:, i, :] = co_solver_l(input_intermediate)  # Apply co-solver for level l, shape (batch_size, dim + 1)
                    input_intermediates[:,i,:]=input_intermediate
                simulated_u, simulated_z = simulated[:, :, 0].reshape(batch_size, MC, 1), simulated[:, :, 1:]  # Extract u and z values, shapes (batch_size, MC, 1) and (batch_size, MC, dim)
                y = np.array([f(input_intermediates[:,i,:], simulated_u[:, i, :], simulated_z[:, i, :]) for i in range(MC)])  # Apply generator term function, shape (MC, batch_size, 1)
                y = y.transpose(1, 0, 2)  # Transpose to shape (batch_size, MC, 1)
                u += wloc[:, k, q - 1][:, np.newaxis] * np.mean(y, axis=1)  # Update u values
                # if (cloc[:, k, q - 1] - t).any() == 0:
                #     delta_t = (cloc[:, k, q - 1] - t + 1e-6)[:, np.newaxis]  # Avoid division by zero, shape (batch_size, 1)
                #     z += wloc[:, k, q - 1][:, np.newaxis] * np.sum(y * W, axis=1) / (MC * delta_t)  # Update z values
                # else:
                #     z += wloc[:, k, q - 1][:, np.newaxis] * np.sum(y * W, axis=1) / (MC * (cloc[:, k, q - 1] - t)[:, np.newaxis])  # Update z values
                delta_t = (cloc[:, k, q - 1] - t + 1e-6)[:, np.newaxis]  # Avoid division by zero, shape (batch_size, 1)
                z += wloc[:, k, q - 1][:, np.newaxis] * np.sum(y * W, axis=1) / (MC * delta_t)  # Update z values                
                # Adjust u and z values if l > 0
                if l:
                    input_intermediates=np.zeros((batch_size,MC,dim+1))
                    for i in range(MC):
                        input_intermediate = np.concatenate((X[:, i, :], cloc[:, k, q - 1][:, np.newaxis]), axis=1)  # Intermediate spatial-temporal coordinates, shape (batch_size, n_input)
                        simulated[:, i, :] = co_solver_l_minus_1(input_intermediate)  # Apply co-solver for level l - 1, shape (batch_size, dim + 1)
                        input_intermediates[:,i,:]=input_intermediate
                    simulated_u, simulated_z = simulated[:, :, 0].reshape(batch_size, MC, 1), simulated[:, :, 1:]  # Extract u and z values, shapes (batch_size, MC, 1) and (batch_size, MC, dim)
                    y = np.array([f(input_intermediates[:,i,:], simulated_u[:, i, :], simulated_z[:, i, :]) for i in range(MC)])  # Apply generator term function, shape (MC, batch_size, 1)
                    y = y.transpose(1, 0, 2)  # Transpose to shape (batch_size, MC, 1)
                    u -= wloc[:, k, q - 1][:, np.newaxis] * np.mean(y, axis=1)  # Adjust u values
                    # if (cloc[:, k, q - 1] - t).any() == 0:
                    #     delta_t = (cloc[:, k, q - 1] - t + 1e-6)[:, np.newaxis]  # Avoid division by zero, shape (batch_size, 1)
                    #     z -= wloc[:, k, q - 1][:, np.newaxis] * np.sum(y * W, axis=1) / (MC * delta_t)  # Adjust z values
                    # else:
                    #     z -= wloc[:, k, q - 1][:, np.newaxis] * np.sum(y * W, axis=1) / (MC * (cloc[:, k, q - 1] - t)[:, np.newaxis])  # Adjust z values
                    delta_t = (cloc[:, k, q - 1] - t + 1e-6)[:, np.newaxis]  # Avoid division by zero, shape (batch_size, 1)
                    z -= wloc[:, k, q - 1][:, np.newaxis] * np.sum(y * W, axis=1) / (MC * delta_t)  # Adjust z values
        return np.concatenate((u, z), axis=-1)  # Concatenate adjusted u and z values, shape (batch_size, dim + 1)

    def u_solve(self, n, rho, x_t):
        '''
        Approximate the solution of the PDE, return the value of u(x_t), batchwisely.
        
        Parameters:
            n (int): Number of backward Euler samples needed.
            rho (int): Current level.
            x_t (ndarray): A batch of spatial-temporal coordinates, shape (batch_size, n_input), where
                           batch_size is the number of samples in the batch and n_input is the number of input features (spatial dimensions + 1 for time).
        
        Returns:
            ndarray: The u values for each sample in the batch, shape (batch_size, 1).
                     Here, u is the approximate solution of the PDE at the given coordinates.
        '''
        return self.uz_solve(n, rho, x_t)[:, 0]  # Call uz_solve and return only the u values, shape (batch_size, 1)
   