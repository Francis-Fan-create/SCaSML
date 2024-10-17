import torch
import numpy as np
import deepxde as dde
import torch.nn as nn
from scipy.special import lambertw
from utils.log_variables import log_variables

class ScaSML_full_history(object):
    '''Multilevel Picard Iteration calibrated PINN for high dimensional semilinear PDE'''
    #all the vectors uses rows as index and columns as dimensions
    def __init__(self, equation, net):
        '''
        Initialize the ScaSML parameters.
        
        Parameters:
            equation (Equation): An object representing the equation to be solved.
            net (torch.nn.Module): A PyTorch neural network model for approximating the solution.
        '''
        # Initialize ScaSML parameters from the equation
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
        eq = self.equation
        # batch_size=x_t.shape[0]
        # self.evaluation_counter+=batch_size
        self.evaluation_counter+= 1
        # Convert input data to PyTorch tensor with gradient tracking
        tensor_x_t = torch.tensor(x_t, requires_grad=True).float()
        # Forward pass through the network
        tensor_u_hat = self.net(tensor_x_t) 
        # Convert the network output to numpy array
        u_hat = tensor_u_hat.detach().cpu().numpy()
        # Compute the gradient of the network output with respect to inputs
        tensor_grad_u_hat_x = torch.autograd.grad(tensor_u_hat, tensor_x_t, grad_outputs=torch.ones_like(tensor_u_hat), retain_graph=True, create_graph=True)[0][:, :-1]
        grad_u_hat_x = tensor_grad_u_hat_x.detach().cpu().numpy()
        epsilon=eq.PDE_loss(tensor_x_t,tensor_u_hat,tensor_grad_u_hat_x).detach().cpu().numpy()
        # Calculate the values for the generator function
        '''TO DO: should we multiply z_breve with sigma(x_t) or not?'''
        '''Personally, I think we should, since the W in the algorithm is not multiplied by sigma.'''
        val1 = eq.f(x_t, u_breve + u_hat, eq.sigma(x_t) * (grad_u_hat_x+ z_breve))
        val2 = eq.f(x_t, u_hat, eq.sigma(x_t) * grad_u_hat_x)
        # return val1 - val2
        return val1-val2+epsilon #large version
    
    def g(self, x_t):
        '''
        Terminal constraint function of ScaSML.
        
        Parameters:
            x_t (ndarray): Input data of shape (batch_size, n_input).
        
        Returns:
            ndarray: The output of the terminal constraint function of shape (batch_size,).
        '''
        eq = self.equation
        # batch_size=x_t.shape[0]
        # self.evaluation_counter+=batch_size
        self.evaluation_counter+= 1
        # Convert input data to PyTorch tensor
        tensor_x_t = torch.tensor(x_t, requires_grad=True).float()
        # tensor_x_t[:, -1] = self.T
        # Compute the network output and convert it to numpy array
        u_hat = self.net(tensor_x_t).detach().cpu().numpy()
        # Calculate the result of the terminal constraint function
        result = eq.g(x_t) - u_hat[:, 0]
        # if np.abs(result).any() > 0.5:
        #     print(f'g:{result}')
        return result
    
    def approx_parameters(self, rhomax):
        '''
        Approximates parameters for the multilevel Picard iteration.
        
        Args:
            rhomax (int): Maximum level of refinement.
            
        Returns:
            tuple: A tuple containing matrices for forward Euler steps (Mf), backward Euler steps (Mg).
                Shapes are as follows: Mf, Mg are (rhomax, rhomax).
        '''
        levels = list(range(1, rhomax + 1))  # Level list
        Mf = np.zeros((rhomax, rhomax))  # Initialize matrix for forward Euler steps
        Mg = np.zeros((rhomax, rhomax + 1))  # Initialize matrix for backward Euler steps
        for rho in range(1, rhomax + 1):
            for k in range(1, levels[rho - 1] + 1):
                Mf[rho - 1][k - 1] = round(rho ** (k / 2))  # Compute forward Euler steps
                Mg[rho - 1][k - 1] = round(rho ** (k - 1))  # Compute backward Euler steps
            Mg[rho - 1][rho] = rho ** rho  # Special case for backward Euler steps
        return Mf, Mg
    
    def set_approx_parameters(self, rhomax):
        '''
        Sets the approximation parameters for the multilevel Picard iteration.
        This method should be called before solving the PDE.
        
        Args:
            rhomax (int): Maximum level of refinement.
        '''
        self.Mf, self.Mg = self.approx_parameters(rhomax)  # Set approximation parameters
        
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
        # Set alpha=1/2, beta=1/6
        # Extract model parameters and functions
        Mf, Mg = self.Mf, self.Mg
        T = self.T  # Terminal time
        dim = self.n_input - 1  # Spatial dimensions
        batch_size = x_t.shape[0]  # Batch size
        sigma = self.sigma(x_t)  # Volatility, shape (batch_size, dim)
        mu= self.mu(x_t)  # Drift, shape (batch_size, dim)
        x = x_t[:, :-1]  # Spatial coordinates, shape (batch_size, dim)
        t = x_t[:, -1]  # Temporal coordinates, shape (batch_size,)
        f = self.f  # Generator term function
        g = self.g  # Terminal constraint function
        
        
        
        # Determine the number of Monte Carlo samples for backward Euler
        MC = int(Mg[rho - 1, n])  # Number of Monte Carlo samples, scalar
        
        # Generate Monte Carlo samples for backward Euler
        W = np.sqrt(T-t)[:, np.newaxis, np.newaxis] * np.random.normal(size=(batch_size, MC, dim))  # Brownian increments, shape (batch_size, MC, dim)
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

        delta_t = (T - t + 1e-6)[:, np.newaxis]  # Avoid division by zero, shape (batch_size, 1)
        z = np.sum(differences * W, axis=1) / (MC * delta_t)  # Compute z values, shape (batch_size, dim)        
        # Recursive call for n > 0
        if n == 0:
            # Convert the network output to numpy array
            u_hat = self.net(torch.tensor(x_t, dtype=torch.float32)).detach().cpu().numpy()
            tensor_x_t = torch.tensor(x_t, requires_grad=True).float()
            tensor_u_hat=self.net(tensor_x_t)
            # Compute the gradient of the network output with respect to inputs
            tensor_grad_u_hat_x = torch.autograd.grad(tensor_u_hat, tensor_x_t, grad_outputs=torch.ones_like(tensor_u_hat), retain_graph=True, create_graph=True)[0][:, :-1]
            grad_u_hat_x = tensor_grad_u_hat_x.detach().cpu().numpy()    
            initial_value= np.concatenate((u_hat, grad_u_hat_x), axis=-1)        
            return np.concatenate((u, z), axis=-1)+initial_value 
        elif n < 0:
            return np.concatenate((u, z), axis=-1)
        # Recursive computation for n > 0
        for l in range(n):
            MC = int(Mf[rho - 1, n - l - 1])  # Number of Monte Carlo samples, scalar
            # Sample a random variable tau in (0,1) with density rho(s) = 1/(2*sqrt(s))
            tau = np.random.uniform(0, 1, size=(batch_size, MC)) ** 2
            # Multiply tau by (T-t)
            sampled_time_steps = (tau * (T-t)[:, np.newaxis]).reshape((batch_size, MC, 1))  # Sample time steps, shape (batch_size, MC)
            X = np.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC, axis=1)  # Replicated spatial coordinates, shape (batch_size, MC, dim)
            W = np.zeros((batch_size, MC, dim))  # Initialize Brownian increments, shape (batch_size, MC, dim)
            simulated = np.zeros((batch_size, MC, dim + 1))  # Initialize array for simulated values, shape (batch_size, MC, dim + 1)
        
            dW =np.sqrt(sampled_time_steps) * np.random.normal(size=(batch_size, MC, dim))  # Brownian increments for current time step, shape (batch_size, MC, dim)
            W += dW  # Accumulate Brownian increments
            X += mu*(sampled_time_steps)+sigma * dW  # Update spatial coordinates
            co_solver_l = lambda X_t: self.uz_solve(n=l, rho=rho, x_t=X_t)  # Co-solver for level l
            co_solver_l_minus_1 = lambda X_t: self.uz_solve(n=l - 1, rho=rho, x_t=X_t)  # Co-solver for level l - 1
            input_intermediates=np.zeros((batch_size,MC,dim+1))
            # Compute u and z values for current quadrature point
            for i in range(MC):
                input_intermediate = np.concatenate((X[:, i, :], sampled_time_steps[:,i,:]), axis=1)  # Intermediate spatial-temporal coordinates, shape (batch_size, n_input)
                simulated[:, i, :] = co_solver_l(input_intermediate)  # Apply co-solver for level l, shape (batch_size, dim + 1)
                input_intermediates[:,i,:]=input_intermediate
            simulated_u, simulated_z = simulated[:, :, 0].reshape(batch_size, MC, 1), simulated[:, :, 1:]  # Extract u and z values, shapes (batch_size, MC, 1) and (batch_size, MC, dim)
            y = np.array([f(input_intermediates[:,i,:], simulated_u[:, i, :], simulated_z[:, i, :]) for i in range(MC)])  # Apply generator term function, shape (MC, batch_size, 1)
            y = y.transpose(1, 0, 2)  # Transpose to shape (batch_size, MC, 1)
            u += 2*(T-t)[:,np.newaxis]* np.mean(np.sqrt(tau)[:,:,np.newaxis]*y, axis=1)  # Update u values
            delta_t = (sampled_time_steps + 1e-6)  # Avoid division by zero, shape (batch_size, 1)
            z += 2*(T-t)[:,np.newaxis] * np.mean((np.sqrt(tau)[:,:,np.newaxis]*y * W / (delta_t)),axis=1)  # Update z values                
            # Adjust u and z values if l > 0
            if l:
                input_intermediates=np.zeros((batch_size,MC,dim+1))
                for i in range(MC):
                    input_intermediate = np.concatenate((X[:, i, :], sampled_time_steps[:,i,:]), axis=1)  # Intermediate spatial-temporal coordinates, shape (batch_size, n_input)
                    simulated[:, i, :] = co_solver_l_minus_1(input_intermediate)  # Apply co-solver for level l, shape (batch_size, dim + 1)
                    input_intermediates[:,i,:]=input_intermediate
                simulated_u, simulated_z = simulated[:, :, 0].reshape(batch_size, MC, 1), simulated[:, :, 1:]  # Extract u and z values, shapes (batch_size, MC, 1) and (batch_size, MC, dim)
                y = np.array([f(input_intermediates[:,i,:], simulated_u[:, i, :], simulated_z[:, i, :]) for i in range(MC)])  # Apply generator term function, shape (MC, batch_size, 1)
                y = y.transpose(1, 0, 2)  # Transpose to shape (batch_size, MC, 1)
                u -= 2*(T-t)[:,np.newaxis]* np.mean(np.sqrt(tau)[:,:,np.newaxis]*y, axis=1)  # Update u values
                delta_t = (sampled_time_steps + 1e-6)  # Avoid division by zero, shape (batch_size, 1)
                z -= 2*(T-t)[:,np.newaxis] * np.mean((np.sqrt(tau)[:,:,np.newaxis]*y * W / (delta_t)),axis=1)  # Update z values  
        return np.concatenate((u, z), axis=-1)  # Concatenate adjusted u and z values, shape (batch_size, dim + 1)

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