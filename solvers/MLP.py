import jax.numpy as jnp
from jax import random
from scipy.special import lambertw

class MLP:
    '''Multilevel Picard Iteration for high dimensional semilinear PDE'''
    # All vectors use rows as index and columns as dimensions
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
        equation.geometry()  # Initialize the geometry related parameters in the equation
        self.T = equation.T  # Terminal time
        self.t0 = equation.t0  # Initial time
        self.n_input = equation.n_input  # Number of input features
        self.n_output = equation.n_output  # Number of output features
        self.evaluation_counter = 0  # Number of evaluations
        self.key = random.PRNGKey(0)  # Random key for JAX

    def f(self, x_t, u, z):
        '''
        Generator function of ScaSML, representing the light and large version.
        
        Parameters:
            x_t (array): Input data of shape (batch_size, n_input).
            u (array): u_true.
            z (array): Gradient of u_true.
        
        Returns:
            array: The output of the generator function of shape (batch_size,).
        '''
        # self.evaluation_counter += 1
        eq = self.equation
        return eq.f(x_t, u, z)
    
    def g(self, x_t):
        '''
        Terminal constraint function of ScaSML.
        
        Parameters:
            x_t (array): Input data of shape (batch_size, n_input).
        
        Returns:
            array: The output of the terminal constraint function of shape (batch_size,).
        '''
        # self.evaluation_counter += 1
        eq = self.equation
        return eq.g(x_t)[:, 0]
    
    def inverse_gamma(self, gamma_input):
        '''
        Compute the inverse of the gamma function for the given input.
        
        Args:
            gamma_input (array): Input values for which to compute the inverse gamma function, shape (n,).
                
        Returns:
            array: The computed inverse gamma values, shape (n,).
        '''
        c = 0.036534  # Constant to avoid singularity
        L = jnp.log((gamma_input + c) / jnp.sqrt(2 * jnp.pi))  # Logarithm part of the inverse gamma function
        return jnp.real(L / jnp.real(lambertw(L / jnp.e)) + 0.5)  # Compute and return the real part

    def lgwt(self, N, a, b):
        '''
        Computes the Legendre-Gauss nodes and weights for numerical integration.
        
        Args:
            N (int): The number of nodes and weights to compute.
            a (float): The lower bound of the integration interval.
            b (float): The upper bound of the integration interval.
                
        Returns:
            tuple: A tuple containing two arrays. The first array contains the nodes (shape: (N,)),
                and the second array contains the weights (shape: (N,)).
        '''
        N -= 1  # Adjust N for zero-based indexing
        N1, N2 = N + 1, N + 2  # Adjusted counts for nodes and weights
        xu = jnp.linspace(-1, 1, N1).reshape(1, -1)  # Initial uniform nodes on [-1, 1], reshaped to row vector
        # Initial guess for nodes using cosine transformation and correction term
        y = jnp.cos((2 * jnp.arange(0, N + 1, 1) + 1) * jnp.pi / (2 * N + 2)) + (0.27 / N1) * jnp.sin(jnp.pi * xu * N / N2)
        L = jnp.zeros((N1, N2))  # Initialize Legendre-Gauss Vandermonde Matrix
        Lp = jnp.zeros((N1, N2))  # Initialize derivative of LG Vandermonde Matrix
        y0 = 2  # Initial value for iteration comparison
        # Iterative computation using Newton-Raphson method
        eps = 2.2204e-16
        iteration = 0
        max_iter = 100
        while jnp.max(jnp.abs(y - y0)) > eps and iteration < max_iter:
            L = L.at[:, 0].set(1)
            Lp = Lp.at[:, 0].set(0)
            L = L.at[:, 1].set(y[0,0])
            Lp = Lp.at[:, 1].set(1)
            for k in range(2, N1 + 1):
                L = L.at[:, k].set((((2 * k - 1) * y * L[:, k - 1] - (k - 1) * L[:, k - 2]) / k)[0])
            Lp = (N2) * (L[:, N1 - 1] - y * L[:, N2 - 1]) / (1 - y * y)
            y0 = y
            y = y0 - L[:, N2 - 1] / Lp
            iteration += 1
        x = (a * (1 - y) + b * (1 + y)) / 2  # Map nodes to [a, b]
        w = (b - a) / ((1 - y * y) * (Lp * Lp)) * (N2 * N2) / (N1 * N1)  # Compute weights
        return x[0], w[0]

    def approx_parameters(self, rhomax):
        '''
        Approximates parameters for the multilevel Picard iteration.
        
        Args:
            rhomax (int): Maximum level of refinement.
                
        Returns:
            tuple: A tuple containing matrices for forward Euler steps (Mf), backward Euler steps (Mg),
                number of quadrature points (Q), quadrature points (c), and quadrature weights (w).
        '''
        levels = list(range(1, rhomax + 1))  # Level list
        Q = jnp.zeros((rhomax, rhomax), dtype=int)  # Initialize matrix for number of quadrature points
        Mf = jnp.zeros((rhomax, rhomax), dtype=int)  # Initialize matrix for forward Euler steps
        Mg = jnp.zeros((rhomax, rhomax + 1), dtype=int)  # Initialize matrix for backward Euler steps
        for rho in range(1, rhomax + 1):
            for k in range(1, levels[rho - 1] + 1):
                Q = Q.at[rho - 1, k - 1].set(int(jnp.round(self.inverse_gamma(rho ** (k / 2)))))
                Mf = Mf.at[rho - 1, k - 1].set(int(jnp.round(rho ** (k / 2))))
                Mg = Mg.at[rho - 1, k - 1].set(int(jnp.round(rho ** (k - 1))))
            Mg = Mg.at[rho - 1, rho].set(rho ** rho)
        qmax = int(jnp.max(Q))
        c = jnp.zeros((qmax, qmax))
        w = jnp.zeros((qmax, qmax))
        for k in range(1, qmax + 1):
            ctemp, wtemp = self.lgwt(k, 0, self.T)
            c = c.at[:, k - 1].set(jnp.concatenate([ctemp[::-1], jnp.zeros(qmax - k)]))
            w =   w.at[:, k - 1].set(jnp.concatenate([wtemp[::-1], jnp.zeros(qmax - k)]))
        return Mf, Mg, Q, c, w

    def uz_solve(self, n, rho, x_t):
        '''
        Approximate the solution of the PDE, return the value of u(x_t) and z(x_t), batch-wise.
        
        Parameters:
            n (int): Current level.
            rho (int): Number of quadrature points.
            x_t (array): A batch of spatial-temporal coordinates, shape (batch_size, n_input).
        
        Returns:
            array: The concatenated u and z values for each sample in the batch, shape (batch_size, 1 + n_input - 1).
        '''
        # Extract model parameters and functions
        self.Mf, self.Mg, self.Q, self.c, self.w = self.approx_parameters(rho)  # Set approximation parameters
        Mf, Mg, Q, c, w = self.Mf, self.Mg, self.Q, self.c, self.w
        T = self.T  # Terminal time
        dim = self.n_input - 1  # Spatial dimensions
        batch_size = x_t.shape[0]  # Batch size
        sigma = self.sigma(x_t)  # Volatility, scalar
        mu = self.mu(x_t)  # Drift, scalar
        x = x_t[:, :-1]  # Spatial coordinates, shape (batch_size, dim)
        t = x_t[:, -1]  # Temporal coordinates, shape (batch_size,)
        g = self.g
        f = self.f

        # Manage random keys of JAX
        key = random.PRNGKey(0)  # Random key for generating Monte Carlo samples
        subkey = random.split(key, 1)[0]  # Subkey for generating Brownian increments

        # Compute local time and weights
        cloc = (T - t)[:, jnp.newaxis, jnp.newaxis] * c[jnp.newaxis, :] / T + t[:, jnp.newaxis, jnp.newaxis]  # Local time, shape (batch_size, 1, 1)
        wloc = (T - t)[:, jnp.newaxis, jnp.newaxis] * w[jnp.newaxis, :] / T  # Local weights, shape (batch_size, 1)

        # Determine the number of Monte Carlo samples for backward Euler
        MC_g = int(Mg[rho - 1, n])  # Number of Monte Carlo samples

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
            q = int(Q[rho - 1, n - l - 1])  # Number of quadrature points, scalar
            d = cloc[:, :q, q - 1] - jnp.concatenate((t[:, jnp.newaxis], cloc[:, :q - 1, q - 1]), axis=1)  # Time steps, shape (batch_size, q)
            MC_f = int(Mf[rho - 1, n - l - 1])  # Number of Monte Carlo samples, scalar

            X = jnp.repeat(x[:, jnp.newaxis, :], MC_f, axis=1)  # Replicated spatial coordinates
            W = jnp.zeros((batch_size, MC_f, dim))  # Initialize Brownian increments

            # Compute simulated values for each quadrature point
            for k in range(q):
                self.key, subkey = random.split(self.key)
                dW_random = random.normal(subkey, shape=(batch_size, MC_f, dim), dtype=jnp.float32)
                dW = jnp.sqrt(d[:, k])[:, jnp.newaxis, jnp.newaxis] * dW_random
                # self.evaluation_counter += MC_f * dim
                W += dW
                X += mu * d[:, k][:, jnp.newaxis, jnp.newaxis] + sigma * dW

                # Define Co-solvers
                co_solver_l = lambda X_t: self.uz_solve(n=l, rho=rho, x_t=X_t)  # Co-solver for level l
                co_solver_l_minus_1 = lambda X_t: self.uz_solve(n=l - 1, rho=rho, x_t=X_t)  # Co-solver for level l - 1

                # Prepare intermediate inputs
                input_intermediates = jnp.concatenate((X, jnp.repeat(cloc[:, k, q - 1][:, jnp.newaxis,jnp.newaxis],MC_f,axis=1)), axis=2)
                input_intermediates_flat = input_intermediates.reshape(-1, self.n_input)

                # Co-solver for level l
                simulated_flat = co_solver_l(input_intermediates_flat)
                simulated = simulated_flat.reshape(batch_size, MC_f, -1)
                simulated_u = simulated[:, :, 0].reshape(batch_size, MC_f, 1)
                simulated_z = simulated[:, :, 1:]

                # Flatten for function evaluation
                simulated_u_flat = simulated_u.reshape(-1,1)
                simulated_z_flat = simulated_z.reshape(-1, dim)
                y_flat = f(input_intermediates_flat, simulated_u_flat, simulated_z_flat)
                self.evaluation_counter+=MC_f
                y = y_flat.reshape(batch_size, MC_f, 1)

                u += wloc[:, k,q-1][:, jnp.newaxis] * jnp.mean(y, axis=1)
                z += wloc[:, k,q-1][:, jnp.newaxis] * jnp.sum(y * W, axis=1) / (MC_f * delta_t)

                # Adjust u and z values if l > 0
                if l :

                    # Prepare intermediate inputs
                    input_intermediates = jnp.concatenate((X, jnp.repeat(cloc[:, k, q-1][:, jnp.newaxis, jnp.newaxis],MC_f,axis=1)), axis=2)
                    input_intermediates_flat = input_intermediates.reshape(-1, self.n_input)
                    # Co-solver for level l
                    simulated_flat = co_solver_l_minus_1(input_intermediates_flat)
                    simulated = simulated_flat.reshape(batch_size, MC_f, -1)
                    simulated_u = simulated[:, :, 0].reshape(batch_size, MC_f, 1)
                    simulated_z = simulated[:, :, 1:]

                    simulated_u_flat = simulated_u.reshape(-1, 1)
                    simulated_z_flat = simulated_z.reshape(-1, dim)
                    y_flat = f(input_intermediates_flat, simulated_u_flat, simulated_z_flat)
                    self.evaluation_counter+=MC_f
                    y = y_flat.reshape(batch_size, MC_f, 1)

                    u -= wloc[:, k, q - 1][:, jnp.newaxis] * jnp.mean(y, axis=1)  # Adjust u values
                    delta_t = (cloc[:, k, q - 1] - t + 1e-6)[:, jnp.newaxis]  # Avoid division by zero, shape (batch_size, 1)
                    z -= wloc[:, k, q - 1][:, jnp.newaxis] * jnp.sum(y * W, axis=1) / (MC_f * delta_t)  # Adjust z values
        output_cated = jnp.concatenate((u, z), axis=-1)  # Concatenate adjusted u and z values, shape (batch_size, dim + 1)
        norm_estimation = self.equation.norm_estimation
        return jnp.clip(output_cated, -norm_estimation, norm_estimation).astype(jnp.float32)  # Clip the output to avoid numerical instability

    def u_solve(self, n, rho, x_t):
        '''
        Approximate the solution of the PDE, return the value of u(x_t), batch-wise.
        
        Parameters:
            n (int): Index of summands in quadratic sum.
            rho (int): Current level.
            x_t (array): A batch of spatial-temporal coordinates, shape (batch_size, n_input).
        
        Returns:
            array: The u values for each sample in the batch, shape (batch_size, 1).
        '''
        return self.uz_solve(n, rho, x_t)[:, 0][:,jnp.newaxis]  # Return only the u values
   