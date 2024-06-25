import torch
import numpy as np
import deepxde as dde
import torch.nn as nn
from scipy.special import lambertw
from utils.log_variables import log_variables

class ScaML(object):
    '''Multilevel Picard Iteration calibrated PINN for high dimensional semilinear PDE'''
    #all the vectors uses rows as index and columns as dimensions
    def __init__(self, equation, net):
        '''
        Initialize the ScaML parameters.
        
        Parameters:
            equation (Equation): An object representing the equation to be solved.
            net (torch.nn.Module): A PyTorch neural network model for approximating the solution.
        '''
        # Initialize ScaML parameters from the equation
        self.equation = equation
        self.sigma = equation.sigma
        self.mu = equation.mu
        equation.geometry()
        self.T = equation.T
        self.t0 = equation.t0
        self.n_input = equation.n_input
        self.n_output = equation.n_output
        # Set the network to evaluation mode
        net.eval()
        self.net = net
        # Note: A potential way to accelerate the inference process is to use a discretized version of the Laplacian.
     
    def f(self, x_t, u_breve, z_breve):
        '''
        Generator function of ScaML, representing the light and large version.
        
        Parameters:
            x_t (ndarray): Input data of shape (batch_size, n_input).
            u_breve (ndarray): approximated u_ture-u_hat.
            z_breve (ndarray): approximated gradient of u_breve.
        
        Returns:
            ndarray: The output of the generator function of shape (batch_size,).
        '''
        eq = self.equation
        # Convert input data to PyTorch tensor with gradient tracking
        tensor_x_t = torch.tensor(x_t, requires_grad=True).float()
        # Forward pass through the network
        tensor_u_hat = self.net(tensor_x_t) 
        # Convert the network output to numpy array
        u_hat = tensor_u_hat.detach().cpu().numpy()
        # Compute the gradient of the network output with respect to inputs
        tensor_grad_u_hat_x = torch.autograd.grad(tensor_u_hat, tensor_x_t, grad_outputs=torch.ones_like(tensor_u_hat), retain_graph=True, create_graph=True)[0][:, :-1]
        grad_u_hat_x = tensor_grad_u_hat_x.detach().cpu().numpy()
        # epsilon=eq.PDE_loss(tensor_x_t,tensor_u_hat,tensor_grad_u_hat_x).detach().cpu().numpy()
        # Calculate the values for the generator function
        val1 = eq.f(x_t, u_breve + u_hat, eq.sigma(x_t) * (grad_u_hat_x + z_breve))
        val2 = eq.f(x_t, u_hat, eq.sigma(x_t) * grad_u_hat_x)
        # Return the difference between val1 and val2 (light version, which does not include epsilon here)
        return val1 - val2
        # return val1-val2-epsilon #large version
    
    def g(self, x_t):
        '''
        Terminal constraint function of ScaML.
        
        Parameters:
            x_t (ndarray): Input data of shape (batch_size, n_input).
        
        Returns:
            ndarray: The output of the terminal constraint function of shape (batch_size,).
        '''
        eq = self.equation
        # Convert input data to PyTorch tensor
        tensor_x_t = torch.tensor(x_t, requires_grad=True).float()
        # tensor_x_t[:, -1] = self.T
        # Compute the network output and convert it to numpy array
        u_hat = self.net(tensor_x_t).detach().cpu().numpy()
        # Calculate the result of the terminal constraint function
        result = eq.g(x_t) - u_hat[:, 0]
        return result
    
    def inverse_gamma(self, gamma_input):
        '''
        Computes the inverse of the gamma function for a given input.
        
        Parameters:
            gamma_input (ndarray): Input array of shape (any,).
        
        Returns:
            ndarray: The computed inverse gamma values of shape (any,).
        '''
        # inverse gamma function
        c = 0.036534 # avoid singularity
        L = np.log((gamma_input+c) / np.sqrt(2 * np.pi)) 
        return np.real(L / lambertw(L / np.e) + 0.5) # inverse gamma function
    
    def lgwt(self, N, a, b):
        '''
        Computes the Legendre-Gauss nodes and weights for numerical integration.
        
        Parameters:
            N (int): Number of nodes.
            a (float): Lower bound of the interval.
            b (float): Upper bound of the interval.
        
        Returns:
            tuple: Two ndarrays, the first of shape (N,) containing the nodes, and the second of shape (N,) containing the weights.
        '''
        # Legendre-Gauss nodes and weights
        N -= 1 # truncation number
        N1, N2 = N+1, N+2 # number of nodes and weights
        xu = np.linspace(-1, 1, N1).reshape(1,-1) # uniform on [-1, 1], and transpose to row vector
        y = np.cos((2 * np.arange(0, N+1, 1)+ 1) * np.pi / (2 * N + 2))+(0.27/N1) * np.sin(np.pi * xu * N / N2) # initial guess
        L = np.zeros((N1, N2)) # Legendre-Gauss Vandermonde Matrix
        Lp = np.zeros((N1, N2)) # Derivative of Legendre-Gauss Vandermonde Matrix
        y0 = 2 
        # compute the zeros of the N+1 Legendre Polynomial
        # using the recursion relation and the Newton-Raphson method
        while np.max(np.abs(y-y0)) > 2.2204e-16: # iterate until new points are uniformly within epsilon of old points
            L[:, 0] = 1 
            Lp[:, 0] = 0
            L[:, 1] = y 
            Lp[:, 1] = 1
            for k in range(2, N1+1):
                L[:, k] = ((2 * k -1)* y * L[:,k-1]-(k-1)*L[:, k-2]) / k
            Lp = (N2) * (L[:, N1-1]-y * L[:, N2-1])/(1-y * y)
            y0 = y
            y = y0 - L[:, N2-1] / Lp
        x = (a * (1-y) + b * (1+y)) / 2 # linear map from [-1, 1] to [a, b]
        w = (b-a) / ((1-y*y) * Lp * Lp) * N2 * N2 / (N1 * N1) # compute weights
        return x[0], w[0]

    def approx_parameters(self, rhomax):
        '''
        Approximates parameters for the MLP based on the maximum level of refinement.
        
        Parameters:
            rhomax (int): Maximum level of refinement.
        
        Returns:
            tuple: Five ndarrays, Mf of shape (rhomax, rhomax), Mg of shape (rhomax, rhomax+1), Q of shape (rhomax, rhomax), c of shape (qmax, qmax), and w of shape (qmax, qmax), where qmax is the maximum number of quadrature points across all levels.
        '''
        # approximate parameters for the MLP
        levels = list(range(1, rhomax+1)) # level list
        Q = np.zeros((rhomax, rhomax)) # number of quadrature points
        Mf = np.zeros((rhomax, rhomax)) # number of forward Euler steps
        Mg = np.zeros((rhomax, rhomax+1)) # number of backward Euler steps
        for rho in range(1, rhomax+1):
            for k in range(1, levels[rho-1]+1):
                Q[rho-1][k-1] = round(self.inverse_gamma(rho ** (k/2))) # inverse gamma function
                Mf[rho-1][k-1] = round(rho ** (k/2)) # forward Euler steps
                Mg[rho-1][k-1] = round(rho ** (k-1)) # backward Euler steps
            Mg[rho-1][rho] = rho ** rho # backward Euler steps
        qmax = int(np.max(Q)) # maximum number of quadrature points
        c = np.zeros((qmax, qmax)) # quadrature points
        w = np.zeros((qmax, qmax)) # quadrature weights
        for k in range(1, qmax+1):
            ctemp, wtemp = self.lgwt(k, 0, self.T) # Legendre-Gauss nodes and weights
            c[:, k-1] = np.concatenate([ctemp[::-1], np.zeros(qmax-k)]) # quadrature points
            w[:, k-1] = np.concatenate([wtemp[::-1], np.zeros(qmax-k)]) # quadrature weights
        return Mf, Mg, Q, c, w
    def set_approx_parameters(self, rhomax):
        '''
        Sets the approximation parameters based on the maximum level of refinement.
        
        Parameters:
            rhomax (int): Maximum level of refinement.
        '''
        # set the approximation parameters
        self.Mf, self.Mg, self.Q, self.c, self.w = self.approx_parameters(rhomax)
        
    # @log_variables
    def uz_solve(self, n, rho, x_t):
        '''
        Approximate the solution of the PDE, return the ndarray of u(x_t) and z(x_t) batchwisely.
        
        Parameters:
            n (int): The number of backward Euler samples needed.
            rho (int): The current level.
            x_t (ndarray): A batch of spatial-temporal coordinates, shape (batch_size, n_input).
            
        Returns:
            ndarray: The concatenated u and z values, shape (batch_size, 1+n_input-1).
        '''
        # Extract model parameters and dimensions
        Mf, Mg, Q, c, w = self.Mf, self.Mg, self.Q, self.c, self.w
        T = self.T  # Terminal time
        dim = self.n_input - 1  # Spatial dimensions
        batch_size = x_t.shape[0]  # Batch size
        sigma = self.sigma(x_t)  # Volatility, shape (batch_size, dim)
        x = x_t[:, :-1]  # Spatial coordinates, shape (batch_size, dim)
        t = x_t[:, -1]  # Temporal coordinates, shape (batch_size,)
        f = self.f  # Generator term
        g = self.g  # Terminal constraint
        cloc = (T - t)[:, np.newaxis, np.newaxis] * c[np.newaxis, :] / T + t[:, np.newaxis, np.newaxis]  # Local time, shape (batch_size, 1, 1)
        wloc = (T - t)[:, np.newaxis, np.newaxis] * w[np.newaxis, :] / T  # Local weights, shape (batch_size, 1, 1)
        MC = int(Mg[rho - 1, n])  # Number of Monte Carlo samples for backward Euler
        
        # Monte Carlo simulation
        W = np.sqrt(T - t)[:, np.newaxis, np.newaxis] * np.random.normal(size=(batch_size, MC, dim))
        X = np.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC, axis=1)
        disturbed_X = X + sigma * W  # Disturbed spatial coordinates, shape (batch_size, MC, dim)
        
        # Initialize arrays for terminal and difference calculations
        terminals = np.zeros((batch_size, MC, 1))
        differences = np.zeros((batch_size, MC, 1))
        
        # Calculate terminals and differences
        for i in range(MC):
            input_terminal = np.concatenate((X[:, i, :], np.full((batch_size, 1), T)), axis=1)
            disturbed_input_terminal = np.concatenate((disturbed_X[:, i, :], np.full((batch_size, 1), T)), axis=1)
            terminals[:, i, :] = g(input_terminal)[:, np.newaxis]
            differences[:, i, :] = (g(disturbed_input_terminal) - g(input_terminal))[:, np.newaxis]
        
        # Calculate u and z
        u = np.mean(differences + terminals, axis=1)
        if (T - t).any() == 0:
            delta_t = (T - t + 1e-6)[:, np.newaxis]
            z = np.sum(differences * W, axis=1) / (MC * delta_t)
        else:
            z = np.sum(differences * W, axis=1) / (MC * (T - t)[:, np.newaxis])
        
        # Recursive calculation for n > 0
        if n <= 0:
            return np.concatenate((u, z), axis=-1)
        for l in range(n):
            q = int(Q[rho - 1, n - l - 1])  # Number of quadrature points
            d = cloc[:, :q, q - 1] - np.concatenate((t[:, np.newaxis], cloc[:, :q - 1, q - 1]), axis=1)  # Time step, shape (batch_size, q)
            MC = int(Mf[rho - 1, n - l - 1])
            X = np.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC, axis=1)
            W = np.zeros((batch_size, MC, dim))
            simulated = np.zeros((batch_size, MC, dim + 1))
            
            # Simulate and calculate u, z for each quadrature point
            for k in range(q):
                dW = np.sqrt(d[:, k])[:, np.newaxis, np.newaxis] * np.random.normal(size=(batch_size, MC, dim))
                W += dW
                X += sigma * dW
                co_solver_l = lambda X_t: self.uz_solve(n=l, rho=rho, x_t=X_t)
                co_solver_l_minus_1 = lambda X_t: self.uz_solve(n=l - 1, rho=rho, x_t=X_t)
                input_intermediates = np.zeros((batch_size, MC, dim + 1))
                
                for i in range(MC):
                    input_intermediate = np.concatenate((X[:, i, :], cloc[:, k, q - 1][:, np.newaxis]), axis=1)
                    simulated[:, i, :] = co_solver_l(input_intermediate)
                    input_intermediates[:, i, :] = input_intermediate
                
                simulated_u, simulated_z = simulated[:, :, 0].reshape(batch_size, MC, 1), simulated[:, :, 1:]
                y = np.array([f(input_intermediates[:, i, :], simulated_u[:, i, :], simulated_z[:, i, :]) for i in range(MC)])
                y = y.transpose(1, 0, 2)
                u += wloc[:, k, q - 1][:, np.newaxis] * np.mean(y, axis=1)
                if (cloc[:, k, q - 1] - t).any() == 0:
                    delta_t = (cloc[:, k, q - 1] - t + 1e-6)[:, np.newaxis]
                    z += wloc[:, k, q - 1][:, np.newaxis] * np.sum(y * W, axis=1) / (MC * delta_t)
                else:
                    z += wloc[:, k, q - 1][:, np.newaxis] * np.sum(y * W, axis=1) / (MC * (cloc[:, k, q - 1] - t)[:, np.newaxis])
                
                # Adjust u, z based on previous level if l > 0
                if l:
                    input_intermediates = np.zeros((batch_size, MC, dim + 1))
                    for i in range(MC):
                        input_intermediate = np.concatenate((X[:, i, :], cloc[:, k, q - 1][:, np.newaxis]), axis=1)
                        simulated[:, i, :] = co_solver_l_minus_1(input_intermediate)
                        input_intermediates[:, i, :] = input_intermediate
                    
                    simulated_u, simulated_z = simulated[:, :, 0].reshape(batch_size, MC, 1), simulated[:, :, 1:]
                    y = np.array([f(input_intermediates[:, i, :], simulated_u[:, i, :], simulated_z[:, i, :]) for i in range(MC)])
                    y = y.transpose(1, 0, 2)
                    u -= wloc[:, k, q - 1][:, np.newaxis] * np.mean(y, axis=1)
                    if (cloc[:, k, q - 1] - t).any() == 0:
                        delta_t = (cloc[:, k, q - 1] - t + 1e-6)[:, np.newaxis]
                        z -= wloc[:, k, q - 1][:, np.newaxis] * np.sum(y * W, axis=1) / (MC * delta_t)
                    else:
                        z -= wloc[:, k, q - 1][:, np.newaxis] * np.sum(y * W, axis=1) / (MC * (cloc[:, k, q - 1] - t)[:, np.newaxis])
        return np.concatenate((u, z), axis=-1)

    def u_solve(self, n, rho, x_t):
        '''
        Approximate the solution of the PDE, return the ndarray of u(x_t) only.
        
        Parameters:
            n (int): The number of backward Euler samples needed.
            rho (int): The current level.
            x_t (ndarray): A batch of spatial-temporal coordinates, shape (batch_size, n_input).
            
        Returns:
            ndarray: The u values, shape (batch_size,1).
        '''
        # Calculate u_breve and z_breve using uz_solve
        u_breve_z_breve = self.uz_solve(n, rho, x_t)
        u_breve, z_breve = u_breve_z_breve[:, 0], u_breve_z_breve[:, 1:]
        
        # Convert x_t to tensor and calculate u_hat using the neural network
        tensor_x_t = torch.tensor(x_t, requires_grad=True).float()
        u_hat = self.net(tensor_x_t).detach().cpu().numpy()[:, 0]
        
        # Calculate and return the final u value
        u = u_breve + u_hat
        return u