import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import time
import sys
import os
import cProfile
import shutil
import copy
from optimizers.Adam import Adam
# L_inf has been removed
import jax.numpy as jnp

class ConvergenceRate(object):
    '''
    Convergence Rate test in high dimensions.

    Attributes:
    equation (object): An object representing the equation to solve.
    dim (int): The dimension of the input space minus one.
    solver1 (object): A jax Gaussian Process model.
    solver2 (object): An object for the MLP solver.
    solver3 (object): An object for the ScaSML solver.
    t0 (float): The initial time.
    T (float): The final time.
    '''
    def __init__(self, equation, solver1, solver2, solver3, is_train):
        '''
        Initializes the converge rate test with given solvers and equation.

        Parameters:
        equation (object): The equation object containing problem specifics.
        solver1 (object): The PINN solver.
        solver2 (object): The MLP solver object.
        solver3 (object): The ScaSML solver object.
        '''
        #save original stdout and stderr
        self.stdout=sys.stdout
        self.stderr=sys.stderr
        # Initialize the parameters
        self.equation = equation
        self.dim = equation.n_input - 1  # equation.n_input: int
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.t0 = equation.t0  # equation.t0: float
        self.T = equation.T  # equation.T: float
        self.is_train = is_train

    def test(self, save_path, rhomax=3, train_iters = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]):
        '''
        Compares solvers on different training iterations.
    
        Parameters:
        save_path (str): The path to save the results.
        opt1 (object): The first optimizer object.
        opt2 (object): The second optimizer object.
        rhomax (int): The fixed value of rho for approximation parameters.
        train_iters (list): The list of training iterations.
        '''
        # Initialize the profiler
        profiler = cProfile.Profile()
        profiler.enable()
        is_train = self.is_train
        # Create the save path if it does not exist
        class_name = self.__class__.__name__
        new_path = f"{save_path}/{class_name}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        save_path = new_path
        directory = f'{save_path}/callbacks'
    
        # Delete former callbacks
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
    
        # Set the approximation parameters
        eq = self.equation
        list_len = len(train_iters)
        error1_list = []
        # error2_list = []
        error3_list = []

        # Generate test data (fixed)
        xt_values_domain, xt_values_boundary = eq.generate_test_data(500, 100 , random='Hammersley')
        xt_values = jnp.concatenate((xt_values_domain, xt_values_boundary), axis=0)
        exact_sol = eq.exact_solution(xt_values)
    
        if is_train:
            for j in range(list_len):
                #train the model
                opt = Adam(eq.n_input,1, self.solver1, eq)
                trained_model1= opt.train(f"{save_path}/model_weights_Adam", iters=train_iters[j])
                self.solver1 = trained_model1
                self.solver3.PINN = trained_model1
                # Predict with solver1
                sol1 = self.solver1.predict(xt_values)
            
                # # Solve with solver2 (baseline solver)
                # sol2 = self.solver2.u_solve(rhomax, rhomax, xt_values)
            
                # Solve with solver3 using the trained solver1
                sol3 = self.solver3.u_solve(rhomax, rhomax, xt_values)
            
                # Compute errors
                errors1 = jnp.linalg.norm(sol1 - exact_sol)
                # errors2 = jnp.linalg.norm(sol2 - exact_sol)
                errors3 = jnp.linalg.norm(sol3 - exact_sol)
            
                error_value1 = errors1 / jnp.linalg.norm(exact_sol)
                # error_value2 = errors2 / jnp.linalg.norm(exact_sol)
                error_value3 = errors3 / jnp.linalg.norm(exact_sol)

                error1_list.append(error_value1)
                # error2_list.append(error_value2)
                error3_list.append(error_value3)
            
            # Plot error ratios
            plt.figure()
            epsilon = 1e-10  # To avoid log(0)

            domain_sizes = jnp.array(train_iters)
            train_sizes = domain_sizes * 2760
            error1_array = jnp.array(error1_list)
            # error2_array = jnp.array(error2_list)
            error3_array = jnp.array(error3_list)

            # plt.plot(train_sizes, error1_array, marker='x', linestyle='-', label='PINN')
            # # plt.plot(train_sizes, error2_array, marker='x', linestyle='-', label='MLP')
            # plt.plot(train_sizes, error3_array, marker='x', linestyle='-', label='ScaSML')
            
            # Fit lines to compute slopes
            log_GN_steps = jnp.log10(train_sizes + epsilon)
            log_error1 = jnp.log10(error1_array+ epsilon)
            # log_error2 = jnp.log10(error2_array+ epsilon)
            log_error3 = jnp.log10(error3_array+ epsilon) 
            slope1, intercept1 = jnp.polyfit(log_GN_steps, log_error1, 1)
            # slope2, intercept2 = jnp.polyfit(log_GN_steps, log_error2, 1)
            slope3, intercept3 = jnp.polyfit(log_GN_steps, log_error3, 1)
            fitted_line1 = 10 ** (intercept1 + slope1 * log_GN_steps)
            # fitted_line2 = 10 ** (intercept2 + slope2 * log_GN_steps)
            fitted_line3 = 10 ** (intercept3 + slope3 * log_GN_steps)
            
            plt.plot(train_sizes, fitted_line1, marker='x', linestyle='--', label=f'PINN: slope={slope1:.2f}')
            # plt.plot(train_sizes, fitted_line2, marker='x', linestyle='--', label=f'MLP: slope={slope2:.2f}')
            plt.plot(train_sizes, fitted_line3, marker='x', linestyle='--', label=f'SCaSML: slope={slope3:.2f}')

            plt.yscale('log')

            plt.legend()
            plt.grid(True, which="both", ls="--", linewidth=0.5)
            
            plt.xlabel('Training Size')
            plt.ylabel('Mean Relative L2 Error on Test Set')
            plt.title('ConvergenceRate Verification')

            plt.tight_layout()
            
            plt.savefig(f'{save_path}/ConvergenceRate_Verification.png')
            plt.close()
        
            # Disable the profiler and print stats
            profiler.disable()
            profiler.print_stats(sort='cumtime')
            is_train = False
            return rhomax
        else:
            print("Please delete the model weights and run the test again.")
