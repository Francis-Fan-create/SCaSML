import jax.numpy as jnp
from jax import random
from scipy.special import lambertw
import jax

class ScaSML:
    '''Multilevel Picard Iteration calibrated PINN for high dimensional semilinear PDE'''

    def __init__(self, equation, PINN):
        '''
        Initialize the ScaSML parameters.

        Parameters:
            equation (Equation): An object representing the equation to be solved.
            PINN (PINN Solver): An object of PINN Solver for PDE.
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
        self.PINN = PINN
        self.evaluation_counter = 0  # Number of evaluations
        self.key = random.PRNGKey(0)  # Random key for JAX

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
        self.evaluation_counter+=1
        u_hat = self.PINN(x_t)
        grad_u_hat_x = jax.grad(lambda tmp: jnp.sum(tmp))(x_t)[:, :-1]
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
        self.evaluation_counter+=1
        u_hat = self.PINN(x_t)
        # tensor_x_t[:, -1] = self.T
        # Calculate the result of the terminal constraint function
        result = eq.g(x_t) - u_hat
        # if jnp.abs(result).any() > 0.5:
        #     print(f'g:{result}')
        return result[:,0]
    
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
        Approximate the solution of the PDE, return u(x_t) and z(x_t).

        Parameters:
            n (int): Current level.
            rho (int): Number of quadrature points.
            x_t (array): Spatial-temporal coordinates, shape (batch_size, n_input).

        Returns:
            array: Concatenated u and z values.
        '''
        self.Mf, self.Mg, self.Q, self.c, self.w = self.approx_parameters(rho)  # Set approximation parameters
        Mf, Mg, Q, c, w = self.Mf, self.Mg, self.Q, self.c, self.w
        eq = self.equation
        T = self.T
        dim = self.n_input - 1
        batch_size = x_t.shape[0]
        sigma = self.sigma(x_t)
        mu = self.mu(x_t)
        x = x_t[:, :-1]
        t = x_t[:, -1]
        f = self.f
        g = self.g

        cloc = (T - t)[:, jnp.newaxis, jnp.newaxis] * c[jnp.newaxis, :] / T + t[:, jnp.newaxis, jnp.newaxis]  # Local time, shape (batch_size, 1, 1)
        wloc = (T - t)[:, jnp.newaxis, jnp.newaxis] * w[jnp.newaxis, :] / T  # Local weights, shape (batch_size, 1, 1)

        MC = int(Mg[rho - 1, n])

        self.key, subkey = random.split(self.key)
        W_random = random.normal(subkey, shape=(batch_size, MC, dim))
        # Monte Carlo simulation
        W = jnp.sqrt(T - t)[:, jnp.newaxis, jnp.newaxis] * W_random
        self.evaluation_counter+=MC
        X = jnp.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC, axis=1)
        disturbed_X = X + mu*(T-t)[:, jnp.newaxis, jnp.newaxis]+ sigma * W  # Disturbed spatial coordinates, shape (batch_size, MC, dim)

        input_terminal = jnp.concatenate((X, jnp.full((batch_size, MC, 1), T)), axis=2)
        disturbed_input_terminal = jnp.concatenate((disturbed_X, jnp.full((batch_size, MC, 1), T)), axis=2)

        input_terminal_flat = input_terminal.reshape(-1, self.n_input)
        disturbed_input_terminal_flat = disturbed_input_terminal.reshape(-1, self.n_input)

        terminals_flat = g(input_terminal_flat)
        differences_flat = g(disturbed_input_terminal_flat) - terminals_flat

        terminals = terminals_flat.reshape(batch_size, MC, 1)
        differences = differences_flat.reshape(batch_size, MC, 1)

        u = jnp.mean(differences + terminals, axis=1)
        delta_t = (T - t + 1e-6)[:, jnp.newaxis]
        z = jnp.sum(differences * W, axis=1) / (MC * delta_t)
        cated_uz = jnp.concatenate((u, z), axis=-1)

        # Recursive call for n > 0
        if n == 0:
            batch_size=x_t.shape[0]
            u_hat = self.PINN(x_t)
            grad_u_hat_x = jax.grad(lambda tmp: jnp.sum(tmp))(x_t)[:, :-1]
            initial_value= jnp.concatenate((u_hat, sigma* grad_u_hat_x), axis=-1)        
            return initial_value 
        elif n < 0:
            return jnp.zeros_like(cated_uz)  # Return zeros if n < 0

        for l in range(n):
            q = int(Q[rho - 1, n - l - 1])
            d = cloc[:, :q, q-1] - jnp.concatenate((t[:, jnp.newaxis], cloc[:, :q - 1, q-1]), axis=1)
            MC = int(Mf[rho - 1, n - l - 1])
            X = jnp.repeat(x[:, jnp.newaxis, :], MC, axis=1)
            W = jnp.zeros((batch_size, MC, dim))

            for k in range(q):
                self.key, subkey = random.split(self.key)
                dW_random = random.normal(subkey, shape=(batch_size, MC, dim))
                dW = jnp.sqrt(d[:, k])[:, jnp.newaxis, jnp.newaxis] * dW_random
                self.evaluation_counter += MC * dim
                W += dW
                X += mu * d[:, k][:, jnp.newaxis, jnp.newaxis] + sigma * dW

                co_solver_l = lambda X_t: self.uz_solve(n=l, rho=rho, x_t=X_t)
                co_solver_l_minus_1 = lambda X_t: self.uz_solve(n=l - 1, rho=rho, x_t=X_t)

                input_intermediates = jnp.concatenate((X, jnp.repeat(cloc[:, k, q - 1][:, jnp.newaxis,jnp.newaxis],MC,axis=1)), axis=2)
                input_intermediates_flat = input_intermediates.reshape(-1, self.n_input)

                simulated_flat = co_solver_l(input_intermediates_flat)
                simulated = simulated_flat.reshape(batch_size, MC, -1)
                simulated_u = simulated[:, :, 0].reshape(batch_size, MC, 1)
                simulated_z = simulated[:, :, 1:]

                simulated_u_flat = simulated_u.reshape(-1, 1)
                simulated_z_flat = simulated_z.reshape(-1, dim)
                y_flat = f(input_intermediates_flat, simulated_u_flat, simulated_z_flat)
                y = y_flat.reshape(batch_size, MC, 1)

                u += wloc[:, k, q - 1][:, jnp.newaxis] * jnp.mean(y, axis=1)
                delta_t = (cloc[:, k, q - 1] - t + 1e-6)[:, jnp.newaxis]
                z += wloc[:, k, q - 1][:, jnp.newaxis] * jnp.sum(y * W, axis=1) / (MC * delta_t) 

                if l :
                    input_intermediates = jnp.concatenate((X, jnp.repeat(cloc[:, k, q-1][:, jnp.newaxis, jnp.newaxis],MC,axis=1)), axis=2)
                    input_intermediates_flat = input_intermediates.reshape(-1, self.n_input)

                    simulated_flat = co_solver_l_minus_1(input_intermediates_flat)
                    simulated = simulated_flat.reshape(batch_size, MC, -1)
                    simulated_u = simulated[:, :, 0].reshape(batch_size, MC, 1)
                    simulated_z = simulated[:, :, 1:]

                    simulated_u_flat = simulated_u.reshape(-1, 1)
                    simulated_z_flat = simulated_z.reshape(-1, dim)
                    y_flat = f(input_intermediates_flat, simulated_u_flat, simulated_z_flat)
                    y = y_flat.reshape(batch_size, MC, 1)

                    u -= wloc[:, k, q - 1][:, jnp.newaxis] * jnp.mean(y, axis=1)
                    delta_t = (cloc[:, k, q - 1] - t + 1e-6)[:, jnp.newaxis]
                    z -= wloc[:, k, q - 1][:, jnp.newaxis] * jnp.sum(y * W, axis=1) / (MC * delta_t)
        output_uz = jnp.concatenate((u, z), axis=-1)
        uncertainty = self.equation.uncertainty
        # Clip output_uz to avoid large values
        return jnp.clip(output_uz, -uncertainty, uncertainty)

    def u_solve(self, n, rho, x_t):
        '''
        Approximate the solution of the PDE, return u(x_t) only.

        Parameters:
            n (int): Index of summands in quadratic sum.
            rho (int): Current level.
            x_t (array): Spatial-temporal coordinates.

        Returns:
            array: u values.
        '''
        eq = self.equation
        # Calculate u_breve and z_breve using uz_solve
        u_breve_z_breve = self.uz_solve(n, rho, x_t)
        u_breve = u_breve_z_breve[:, 0][:, jnp.newaxis]
        
        u_hat = self.PINN(x_t)
        
        return u_hat + u_breve