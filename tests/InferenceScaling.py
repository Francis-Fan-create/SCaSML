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
from scipy.stats import t
import matplotlib.ticker as ticker

class InferenceScaling(object):
    '''
    Inference Scaling test in high dimensions.

    Attributes:
    equation (object): An object representing the equation to solve.
    dim (int): The dimension of the input space minus one.
    solver1 (object): A jax Gaussian Process model.
    solver2 (object): An object for the MLP solver.
    solver3 (object): An object for the ScaSML solver.
    t0 (float): The initial time.
    T (float): The final time.
    '''
    def __init__(self, equation, solver1, solver2, solver3):
        '''
        Initializes the inference scaling test with given solvers and equation.

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

    def test(self, save_path, rhomax=3, n_samples=1000):
        '''
        Compares solvers on different training iterations.
    
        Parameters:
        save_path (str): The path to save the results.
        rhomax (int): The fixed value of rho for approximation parameters.
        n_samples (int): The number of samples for testing (test set).
        '''
        # Initialize the profiler
        profiler = cProfile.Profile()
        profiler.enable()
    
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
        eq_dim = eq.n_input - 1
        geom = eq.geometry()
    
    
        # Fix GN_steps
        GN_steps = 2500
        # Build a list for training sizes
        list_len = rhomax
        train_sizes_domain = 2500
        train_sizes_boundary = 100
        eval_counter_list = []
        error1_list = []
        error2_list = []
        error3_list = []
    
        # Generate test data (fixed)
        n_samples_domain = n_samples
        n_samples_boundary = int(n_samples/5)
        xt_values_domain, xt_values_boundary = eq.generate_test_data(n_samples_domain, n_samples_boundary)
        xt_values = np.concatenate((xt_values_domain, xt_values_boundary), axis=0)
        exact_sol = eq.exact_solution(xt_values)

        # Training solver
        print(f"Training solver1 with {train_sizes_domain} domain points and {train_sizes_boundary} boundary points")
        opt = Adam(eq.n_input,1, self.solver1, eq)
        trained_model1= opt.train(f"{save_path}/model_weights_Adam", iters=GN_steps)
        self.solver1 = trained_model1
        self.solver3.PINN = trained_model1

        for j in range(list_len):

            rho = j + 1

            # Print current rho value
            print(f"Current rho value: {rho}")
        
            # Predict with solver1
            sol1 = self.solver1.predict(xt_values)
        
            # # Solve with solver2 (baseline solver)
            sol2 = self.solver2.u_solve(rho, rho, xt_values)
        
            # Solve with solver3 using the trained solver1
            sol3 = self.solver3.u_solve(rho, rho, xt_values)
        
            # Compute errors
            errors1 = np.linalg.norm(sol1 - exact_sol)
            errors2 = np.linalg.norm(sol2 - exact_sol)
            errors3 = np.linalg.norm(sol3 - exact_sol)
        
            error_value1 = errors1 / np.linalg.norm(exact_sol)
            error_value2 = errors2 / np.linalg.norm(exact_sol)
            error_value3 = errors3 / np.linalg.norm(exact_sol)

            error1_list.append(error_value1)
            error2_list.append(error_value2)
            error3_list.append(error_value3)

            eval_counter_list.append(self.solver3.evaluation_counter)

        # Plot error ratios
        plt.figure()
        epsilon = 1e-10  # To avoid log(0)
        
        # Compute error arrays for PINN and MLP, and combine them using element-wise min
        error1_array = np.array(error1_list)
        error2_array = np.array(error2_list)
        error3_array = np.array(error3_list)  # SCaSML
        evaluation_counter_array = np.array(eval_counter_list)
        
        error_min_array = np.minimum(error1_array, error2_array)  # Combined PINN/MLP error
        
        # Compute improvement percentage: 
        # (min(PINN,MLP) - SCaSML) / min(PINN,MLP) * 100, allowing positive or negative values
        improvement = (error_min_array - error3_array) / error_min_array * 100
        
        # Use 'seaborn' style since 'seaborn-whitegrid' is unavailable
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(3.5, 3))  # 3.5 inches ~89mm width
        ax = fig.add_subplot(111)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=4, pad=2)
        
        # Define color for the improvement curve using SCaSML's color
        COLOR_PALETTE = {
            'SCaSML': '#2C939A'
        }
        
        # Plot improvement percentage curve (both line and marker)
        ax.plot(evaluation_counter_array, improvement,
                color=COLOR_PALETTE['SCaSML'], linestyle='-', marker='o',
                linewidth=1.5, markersize=4, label='Improvement (%)')
        
        # Configure axis labels
        ax.set_xlabel('Realization of Normals', labelpad=3)
        ax.set_ylabel('Improvement (%)', labelpad=3)
        
        # Use logarithmic scale for x-axis to mimic scaling law plots
        ax.set_xscale('log')
        
        # Adjust y-axis limits to highlight variation in improvement (using 20% margin)
        y_min = improvement.min()
        y_max = improvement.max()
        y_margin = (y_max - y_min) * 0.2 if (y_max - y_min) != 0 else 1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Add minimalist legend
        ax.legend(frameon=False, loc='upper right', handlelength=1.5, handletextpad=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/InferenceScaling_Improvement.pdf', format='pdf',
                    bbox_inches='tight', pad_inches=0.05)
        plt.close()

        # Disable the profiler and print stats
        profiler.disable()
        profiler.print_stats(sort='cumtime')
    
        return rhomax