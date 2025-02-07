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
        GN_steps = 1000 
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

        error1_array = np.array(error1_list)
        error2_array = np.array(error2_list)
        error3_array = np.array(error3_list)
        evaluation_counter_array = np.array(eval_counter_list)

        # plt.plot(evaluation_counter_array, error1_array, marker='x', linestyle='-', label='PINN')
        # plt.plot(evaluation_counter_array, error2_array, marker='x', linestyle='-', label='MLP')
        # plt.plot(evaluation_counter_array, error3_array, marker='x', linestyle='-', label='ScaSML')
        
        # Fit lines to compute slopes
        log_GN_steps = np.log10(evaluation_counter_array + epsilon)
        log_error1 = np.log10(error1_array+ epsilon)
        log_error2 = np.log10(error2_array+ epsilon)
        log_error3 = np.log10(error3_array+ epsilon) 
        slope1, intercept1 = np.polyfit(log_GN_steps, log_error1, 1)
        slope2, intercept2 = np.polyfit(log_GN_steps, log_error2, 1)
        slope3, intercept3 = np.polyfit(log_GN_steps, log_error3, 1)
        fitted_line1 = 10 ** (intercept1 + slope1 * log_GN_steps)
        fitted_line2 = 10 ** (intercept2 + slope2 * log_GN_steps)
        fitted_line3 = 10 ** (intercept3 + slope3 * log_GN_steps)
        
        # ======================
        # Visualization Settings
        # ======================
        # Define custom color scheme (Black, Gray, Teal)
        COLOR_PALETTE = {
            'PINN': '#000000',    # Primary black
            'MLP': '#A6A3A4',   # Neutral gray
            'SCaSML': '#2C939A' # Scientific teal
        }

        # Configure matplotlib rcParams for publication quality
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',        # Set font family
            'font.size': 9,                # Base font size
            'axes.labelsize': 10,          # Axis label size
            'axes.titlesize': 0,           # Disable title (per request)
            'legend.fontsize': 8,          # Legend font size
            'xtick.labelsize': 8,          # X-tick label size
            'ytick.labelsize': 8,          # Y-tick label size
            'axes.linewidth': 0.8,         # Axis line width
            'lines.linewidth': 1.2,        # Plot line width
            'lines.markersize': 5,         # Marker size
            'savefig.dpi': 600,            # Output resolution
            'savefig.transparent': True,   # Transparent background
            'figure.autolayout': True      # Enable tight layout
        })

        # =========================
        # Confidence Interval Calculation
        # =========================
        def calculate_confidence_interval(log_x, log_y, slope, intercept, alpha=0.95):
            """Calculate 95% confidence interval for regression line"""
            # Calculate predicted values
            log_y_pred = slope * log_x + intercept
            
            # Compute residuals and standard error
            residuals = log_y - log_y_pred
            SSE = np.sum(residuals**2)
            n = len(log_x)
            df = n - 2
            MSE = SSE / df
            x_mean = np.mean(log_x)
            t_crit = t.ppf((1 + alpha)/2, df)  # Critical t-value
            
            # Calculate confidence bands
            se = np.sqrt(MSE * (1/n + (log_x - x_mean)**2 / np.sum((log_x - x_mean)**2)))
            ci_upper = log_y_pred + t_crit * se
            ci_lower = log_y_pred - t_crit * se
            
            return 10**ci_upper, 10**ci_lower

        # Calculate confidence intervals for all methods
        ci_upper1, ci_lower1 = calculate_confidence_interval(log_GN_steps, log_error1, slope1, intercept1)
        ci_upper2, ci_lower2 = calculate_confidence_interval(log_GN_steps, log_error2, slope2, intercept2)
        ci_upper3, ci_lower3 = calculate_confidence_interval(log_GN_steps, log_error3, slope3, intercept3)

        # ======================
        # Figure Composition
        # ======================
        # Create figure with specific dimensions (Nature standard: 89mm single column)
        fig = plt.figure(figsize=(3.5, 3))  # 3.5 inches = ~89mm width
        
        # Configure axis spacing
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=4, pad=2)

        # ======================
        # Data Visualization
        # ======================
        # Plot confidence intervals first
        fill_alpha = 0.15  # Subtle transparency for confidence bands
        for method, ci_upper, ci_lower in zip(['GP', 'MLP', 'SCaSML'],
                                            [ci_upper1, ci_upper2, ci_upper3],
                                            [ci_lower1, ci_lower2, ci_lower3]):
            ax.fill_between(evaluation_counter_array, ci_lower, ci_upper,
                        color=COLOR_PALETTE[method], alpha=fill_alpha, 
                        linewidth=0, zorder=1)
        # Plot regression lines with distinct markers
        marker_params = {
            'GP': {'marker': 'o', 'facecolor': 'none', 'edgewidth': 0.8},
            'MLP': {'marker': 's', 'facecolor': 'none', 'edgewidth': 0.8},
            'SCaSML': {'marker': '^', 'facecolor': 'none', 'edgewidth': 0.8}
        }
        
        # Plot original points
        for method, error_array in zip(['GP', 'MLP', 'SCaSML'],
                                    [error1_array, error2_array, error3_array]):
            ax.scatter(evaluation_counter_array, error_array,
                    color=COLOR_PALETTE[method],
                    marker=marker_params[method]['marker'],
                    s=20,  # Adjust marker size
                    edgecolor=COLOR_PALETTE[method],
                    linewidths=0.5,
                    alpha=0.8,
                    zorder=3)

        # Plot regression lines
        for method, line in zip(['GP', 'MLP', 'SCaSML'],
                            [fitted_line1, fitted_line2, fitted_line3]):
            ax.plot(evaluation_counter_array, line,
                color=COLOR_PALETTE[method],
                linestyle='--',
                linewidth=1.2,
                zorder=2)

        # ======================
        # Aesthetic Refinements
        # ======================
        # Configure axis labels
        ax.set_xlabel('Evaluation Steps', labelpad=3)
        ax.set_ylabel('Relative L2 Error', labelpad=3)

        # Set axis limits and scale
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(left=0)  # Keep linear scale per request

        # Set x-ticks (manually) and rotate the labels
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 6))  
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # Create minimalist legend
        legend_elements = [
            plt.Line2D([0], [0], color=COLOR_PALETTE['GP'], lw=1.2,
                    label=f'GP (m={slope1:.2f})'),
            plt.Line2D([0], [0], color=COLOR_PALETTE['MLP'], lw=1.2,
                    label=f'MLP (m={slope2:.2f})'),
            plt.Line2D([0], [0], color=COLOR_PALETTE['SCaSML'], lw=1.2,
                    label=f'SCaSML (m={slope3:.2f})')
        ]
        ax.legend(handles=legend_elements, frameon=False,
                loc='upper right', bbox_to_anchor=(1, 1),
                handlelength=1.5, handletextpad=0.5)

        # Add gridlines
        ax.grid(True, which='major', axis='y', linestyle='--', 
            linewidth=0.5, alpha=0.4)

        # ======================
        # Output Configuration
        # ======================
        plt.tight_layout()
        plt.savefig(f'{save_path}/InferenceScaling_Verification.pdf',  # Vector format preferred
                format='pdf', bbox_inches='tight', pad_inches=0.05)
        plt.close()
    
        # Disable the profiler and print stats
        profiler.disable()
        profiler.print_stats(sort='cumtime')
    
        return rhomax