import numpy as np
import jax.numpy as jnp
from jax import random
import jax
import deepxde as dde

class ScaSML_full_history(object):
    '''Multilevel Picard Iteration calibrated PINN for high dimensional semilinear PDE'''
    #all the vectors uses rows as index and columns as dimensions
    def __init__(self, equation, PINN):
        '''
        Initialize the ScaSML parameters.
        
        Parameters:
            equation (Equation): An object representing the equation to be solved.
            PINN (dde.Model): An object of PINN Solver for PDE.
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
        self.model = PINN
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
        # self.evaluation_counter+=1
        u_hat = self.model.predict(x_t)
        grad_u_hat_x = self.model.predict(x_t,operator=self.equation.grad)
        # Calculate the values for the generator function
        val1 = eq.f(x_t, u_breve + u_hat, eq.sigma(x_t) * (grad_u_hat_x)+ z_breve)
        val2 = eq.f(x_t, u_hat, eq.sigma(x_t) * grad_u_hat_x)
        return val1 - val2 #light version
    
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
        # self.evaluation_counter+=1
        u_hat = self.model.predict(x_t)
        # tensor_x_t[:, -1] = self.T
        # Calculate the result of the terminal constraint function
        result = eq.g(x_t) - u_hat
        # if jnp.abs(result).any() > 0.5:
        #     print(f'g:{result}')
        return result[:,0]
        
    # @log_variables
    def uz_solve(self, n, rho, x_t, M):
        '''
        Approximate the solution of the PDE, return the value of u(x_t) and z(x_t), batchwisely.
        
        Parameters:
            n (int): Total levels for current solver.
            rho (int): Number of quadrature points, not used in this solver.
            x_t (ndarray): A batch of spatial-temporal coordinates, shape (batch_size, n_input), where
                           batch_size is the number of samples in the batch and n_input is the number of input features (spatial dimensions + 1 for time).
            M (int) : Exponential base for sample size.
        
        Returns:
            ndarray: The concatenated u and z values for each sample in the batch, shape (batch_size, 1+n_input-1).
                     Here, u is the approximate solution of the PDE at the given coordinates, and z is the associated spatial gradient.
        '''
        # Set alpha=1
        # Extract model parameters and functions
        T = self.T  # Terminal time
        dim = self.n_input - 1  # Spatial dimensions
        batch_size = x_t.shape[0]  # Batch size
        sigma = self.sigma(x_t)  # Volatility, scalar
        mu= self.mu(x_t)  # Drift, scalar
        x = x_t[:, :-1]  # Spatial coordinates, shape (batch_size, dim)
        t = x_t[:, -1]  # Temporal coordinates, shape (batch_size,)
        f = self.f  # Generator term function
        g = self.g  # Terminal constraint function
        
        # Manage random keys of JAX
        key = random.PRNGKey(0)  # Random key for generating Monte Carlo samples
        subkey = random.split(key, 1)[0]  # Subkey for generating Brownian increments
        
        # Determine the number of Monte Carlo samples for backward Euler
        MC_g = int(M**n)  # Number of Monte Carlo samples, scalar
        
        # Generate Monte Carlo samples for backward Euler
        std_normal = random.normal(subkey, shape=(batch_size, MC_g, dim), dtype=jnp.float32)
        dW = jnp.sqrt(T-t)[:, jnp.newaxis, jnp.newaxis] * std_normal  # Brownian increments, shape (batch_size, MC_g, dim)
        # self.evaluation_counter+=MC_g
        X = jnp.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC_g, axis=1)  # Replicated spatial coordinates, shape (batch_size, MC_g, dim)
        disturbed_X = X + mu*(T-t)[:, jnp.newaxis, jnp.newaxis]+ sigma * dW  # Disturbed spatial coordinates, shape (batch_size, MC_g, dim)
        
        
        # Prepare terminal inputs
        disturbed_input_terminal = jnp.concatenate((disturbed_X, jnp.full((batch_size, MC_g, 1), T)), axis=2)  # Shape (batch_size, MC_g, n_input)

        # Flatten inputs for vectorized function evaluation
        disturbed_input_terminal_flat = disturbed_input_terminal.reshape(-1, self.n_input)

        # Vectorized evaluation of g
        distrubed_output_terminal_flat = g(disturbed_input_terminal_flat)  # Evaluate disturbed terminal condition, shape (batch_size * MC_g, 1)
        self.evaluation_counter+=MC_g

        # Reshape back to (batch_size, MC_g, 1)
        distrubed_output_terminal =  distrubed_output_terminal_flat.reshape(batch_size, MC_g, 1)
        
        # Compute u and z values
        u = jnp.mean(distrubed_output_terminal, axis=1)  # Mean over Monte Carlo samples, shape (batch_size, 1)

        delta_t = (T - t + 1e-6)[:, jnp.newaxis]  # Avoid division by zero, shape (batch_size, 1)
        z = jnp.mean(distrubed_output_terminal * std_normal, axis=1) / (delta_t)  # Compute z values, shape (batch_size, dim)   
        cated_uz = jnp.concatenate((u, z), axis=-1)  # Concatenate u and z values, shape (batch_size, dim + 1)     
        # Recursive call
        if n == 0:
            initial_output = jnp.zeros_like(cated_uz)
            return initial_output 
        
        # Recursive computation for n > 0
        for l in range(n):
            MC_f = int(M**(n-l))  # Number of Monte Carlo samples, scalar
            tau = random.uniform(subkey, shape=(batch_size, MC_f), dtype=jnp.float32)  # Sample a random variable tau in (0,1) with uniform distribution, shape (batch_size, MC_f)
            # Multiply tau by (T-t)
            sampled_time_steps = (tau * (T-t)[:, jnp.newaxis]).reshape((batch_size, MC_f, 1))  # Sample time steps, shape (batch_size, MC_f)
            X = jnp.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC_f, axis=1)  # Replicated spatial coordinates, shape (batch_size, MC_f, dim)
            simulated = jnp.zeros((batch_size, MC_f, dim + 1))  # Initialize array for simulated values, shape (batch_size, MC_f, dim + 1)
            std_normal = random.normal(subkey, shape=(batch_size, MC_f, dim), dtype=jnp.float32)  # Generate standard normal samples
            dW =jnp.sqrt(sampled_time_steps) * std_normal  # Brownian increments for current time step, shape (batch_size, MC_f, dim)
            # self.evaluation_counter+=MC_f*dim
            X += mu*(sampled_time_steps)+sigma * dW  # Update spatial coordinates
            co_solver_l = lambda X_t: self.uz_solve(n=l, rho= l, x_t=X_t,M=M)  # Co-solver for level l
            co_solver_l_minus_1 = lambda X_t: self.uz_solve(n=l - 1, rho= l, x_t=X_t,M=M)  # Co-solver for level l - 1
            # Compute Compute u and z values for current quadrature point using vmap
            input_intermediates = jnp.concatenate((X, sampled_time_steps), axis=2)  # Intermediate spatial-temporal coordinates, shape (batch_size, MC_f, n_input)
            input_intermediates_flat = input_intermediates.reshape(-1, self.n_input)
            simulated_flat = co_solver_l(input_intermediates_flat)
            simulated = simulated_flat.reshape(batch_size, MC_f, dim + 1)
            simulated_u, simulated_z = simulated[:, :, 0].reshape(batch_size, MC_f, 1), simulated[:, :, 1:]  # Extract u and z values, shapes (batch_size, MC_f, 1) and (batch_size, MC_f, dim)
            # Flatten inputs for vectorized function evaluation and apply generator term function
            simulated_u_flat = simulated_u.reshape(-1, 1)
            simulated_z_flat = simulated_z.reshape(-1, dim)
            y_flat = f(input_intermediates_flat, simulated_u_flat, simulated_z_flat)  # Apply generator term function, shape (batch_size * MC_f, 1)
            self.evaluation_counter+=MC_f
            y = y_flat.reshape(batch_size, MC_f, 1)  # Reshape to shape (batch_size, MC_f, 1)
            # Update u and z values
            u += (T-t)[:,jnp.newaxis]* jnp.mean(y, axis=1)  # Update u values
            delta_sqrt_t = jnp.sqrt(sampled_time_steps + 1e-6)  # Avoid division by zero, shape (batch_size, 1)
            z += (T-t)[:,jnp.newaxis] * jnp.mean((y * std_normal / (delta_sqrt_t)),axis=1)  # Update z values                
            # Adjust u and z values if l > 0
            if l:
                # Compute Compute u and z values for current quadrature point using vmap
                input_intermediates = jnp.concatenate((X, sampled_time_steps), axis=2)  # Intermediate spatial-temporal coordinates, shape (batch_size, MC_f, n_input)
                input_intermediates_flat = input_intermediates.reshape(-1, self.n_input)
                simulated_flat = co_solver_l_minus_1(input_intermediates_flat)
                simulated = simulated_flat.reshape(batch_size, MC_f, dim + 1)
                simulated_u, simulated_z = simulated[:, :, 0].reshape(batch_size, MC_f, 1), simulated[:, :, 1:]  # Extract u and z values, shapes (batch_size, MC_f, 1) and (batch_size, MC_f, dim)
                # Flatten inputs for vectorized function evaluation and apply generator term function
                simulated_u_flat = simulated_u.reshape(-1, 1)
                simulated_z_flat = simulated_z.reshape(-1, dim)
                y_flat = f(input_intermediates_flat, simulated_u_flat, simulated_z_flat) # Apply generator term function, shape (batch_size * MC_f, 1)
                self.evaluation_counter+=MC_f
                y = y_flat.reshape(batch_size, MC_f, 1)  # Reshape to shape (batch_size, MC_f, 1)
                # Update u and z values
                u -= (T-t)[:,jnp.newaxis]* jnp.mean(y, axis=1)  # Update u values
                delta_sqrt_t = jnp.sqrt(sampled_time_steps + 1e-6)  # Avoid division by zero, shape (batch_size, 1)
                z -= (T-t)[:,jnp.newaxis] * jnp.mean((y * std_normal / (delta_sqrt_t)),axis=1)  # Update z values
            else:
                epsilon_flat = self.model.predict(input_intermediates_flat,operator = self.equation.PDE_loss)
                epsilon = epsilon_flat.reshape(batch_size, MC_f, 1)
                # Update u and z values
                u += (T-t)[:,jnp.newaxis]* jnp.mean(epsilon, axis=1)  # Update u values
                delta_sqrt_t = jnp.sqrt(sampled_time_steps + 1e-6)  # Avoid division by zero, shape (batch_size, 1)
                z += (T-t)[:,jnp.newaxis] * jnp.mean((epsilon * std_normal / (delta_sqrt_t)),axis=1)  # Update z values                  
        output_uz = jnp.concatenate((u, z), axis=-1)  # Concatenate u and z values, shape (batch_size, dim + 1)
        uncertainty = self.equation.uncertainty
        return jnp.clip(output_uz, -uncertainty, uncertainty)

    def u_solve(self, n, rho, x_t, M=400):
        '''
        Approximate the solution of the PDE, return the ndarray of u(x_t) only.
        
        Parameters:
            n (int): Index of summands in quadratic sum.
            rho (int): The current level.
            x_t (ndarray): A batch of spatial-temporal coordinates, shape (batch_size, n_input).
            
        Returns:
            ndarray: The u values, shape (batch_size,1).
        '''
        eq = self.equation
        # Calculate u_breve and z_breve using uz_solve
        u_breve_z_breve = self.uz_solve(n, rho, x_t, M)
        u_breve = u_breve_z_breve[:, 0][:, jnp.newaxis]
        
        u_hat = self.model.predict(x_t)
        
        return u_hat + u_breve