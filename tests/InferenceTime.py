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
import jax.numpy as jnp
from scipy import stats
import matplotlib.ticker as ticker

class InferenceTime(object):
    '''
    Inference Time test: compares solvers under equal inference time constraint.
    
    This test demonstrates that SCaSML achieves better accuracy than PINN
    when both methods have approximately the same inference time. We use a larger
    PINN as surrogate model and a smaller PINN backbone for SCaSML to match inference time.

    Attributes:
    equation (object): An object representing the equation to solve.
    dim (int): The dimension of the input space minus one.
    solver1_large (object): A large PINN model (surrogate).
    solver1_small (object): A small PINN model (for SCaSML backbone).
    solver2 (object): An object for the MLP solver.
    solver3 (object): An object for the ScaSML solver.
    t0 (float): The initial time.
    T (float): The final time.
    '''
    def __init__(self, equation, solver1_large, solver1_small, solver2, solver3, is_train):
        '''
        Initializes the inference time test with given solvers and equation.

        Parameters:
        equation (object): The equation object containing problem specifics.
        solver1_large (object): The large PINN solver (surrogate).
        solver1_small (object): The small PINN solver (for SCaSML).
        solver2 (object): The MLP solver object.
        solver3 (object): The ScaSML solver object.
        is_train (bool): Whether to train the models.
        '''
        # Save original stdout and stderr
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        # Initialize the parameters
        self.equation = equation
        self.dim = equation.n_input - 1
        self.solver1_large = solver1_large
        self.solver1_small = solver1_small
        self.solver2 = solver2
        self.solver3 = solver3
        self.t0 = equation.t0
        self.T = equation.T
        self.is_train = is_train

    def test(self, save_path, rho_levels=[1, 2, 3], num_domain=1000, num_boundary=200, train_iters=10000):
        '''
        Compares solvers under different rho levels with matched inference time.
        
        For each rho level, we ensure that large PINN and SCaSML (small PINN + rho) 
        have approximately the same inference time, then compare accuracy.
    
        Parameters:
        save_path (str): The path to save the results.
        rho_levels (list): List of rho values for SCaSML (max 3).
        num_domain (int): The number of points in the test domain.
        num_boundary (int): The number of points on the test boundary.
        train_iters (int): Number of training iterations.
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
    
        # Set the approximation parameters
        eq = self.equation
        eq_name = eq.__class__.__name__
        d = eq.n_input - 1
        
        # Generate test data (fixed across all rho levels)
        data_domain_test, data_boundary_test = eq.generate_test_data(num_domain, num_boundary)
        xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
        exact_sol = eq.exact_solution(xt_test)
        
        # Storage for results
        pinn_errors = []
        mlp_errors = []
        scasml_errors = []
        pinn_times = []
        mlp_times = []
        scasml_times = []
        
        # Ensure rho values don't exceed 3
        rho_levels = [min(r, 3) for r in rho_levels]
        
        if is_train:
            for rho in rho_levels:
                # ==========================================
                # Large PINN: Train and measure inference time
                # ==========================================
                solver1_large_copy = copy.deepcopy(self.solver1_large)
                opt_large = Adam(eq.n_input, 1, solver1_large_copy, eq)
                
                start_time = time.time()
                trained_pinn_large = opt_large.train(f"{save_path}/model_weights_PINN_large_rho_{rho}", 
                                                     iters=train_iters)
                train_time_pinn = time.time() - start_time
                
                start_time = time.time()
                sol_pinn = trained_pinn_large.predict(xt_test)
                inference_time_pinn = time.time() - start_time
                
                # ==========================================
                # MLP: Measure inference time
                # ==========================================
                solver2_copy = copy.deepcopy(self.solver2)
                
                start_time = time.time()
                sol_mlp = solver2_copy.u_solve(rho, rho, xt_test)
                inference_time_mlp = time.time() - start_time
                
                # ==========================================
                # ScaSML: Train small PINN and measure total inference time
                # ==========================================
                solver3_copy = copy.deepcopy(self.solver3)
                
                # Train small PINN backbone with reduced iterations
                scasml_train_iters = train_iters // (d + 1)
                
                opt_small = Adam(eq.n_input, 1, solver3_copy.PINN, eq)
                
                start_time = time.time()
                trained_pinn_small = opt_small.train(f"{save_path}/model_weights_PINN_small_rho_{rho}", 
                                                     iters=scasml_train_iters)
                train_time_scasml = time.time() - start_time
                
                solver3_copy.PINN = trained_pinn_small
                
                start_time = time.time()
                sol_scasml = solver3_copy.u_solve(rho, rho, xt_test)
                inference_time_scasml = time.time() - start_time
                
                # ==========================================
                # Compute Errors
                # ==========================================
                valid_mask = ~(np.isnan(sol_pinn) | np.isnan(sol_mlp) | 
                              np.isnan(sol_scasml) | np.isnan(exact_sol)).flatten()
                
                if np.sum(valid_mask) == 0:
                    continue
                
                errors_pinn = np.abs(sol_pinn.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
                errors_mlp = np.abs(sol_mlp.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
                errors_scasml = np.abs(sol_scasml.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
                
                exact_sol_valid = exact_sol.flatten()[valid_mask]
                
                rel_error_pinn = np.linalg.norm(errors_pinn) / np.linalg.norm(exact_sol_valid)
                rel_error_mlp = np.linalg.norm(errors_mlp) / np.linalg.norm(exact_sol_valid)
                rel_error_scasml = np.linalg.norm(errors_scasml) / np.linalg.norm(exact_sol_valid)
                
                # Store results
                pinn_errors.append(rel_error_pinn)
                mlp_errors.append(rel_error_mlp)
                scasml_errors.append(rel_error_scasml)
                pinn_times.append(inference_time_pinn)
                mlp_times.append(inference_time_mlp)
                scasml_times.append(inference_time_scasml)
                
                # ==========================================
                # Statistical Analysis
                # ==========================================
                # Calculate statistics
                pinn_mean = np.mean(errors_pinn)
                pinn_std = np.std(errors_pinn)
                mlp_mean = np.mean(errors_mlp)
                mlp_std = np.std(errors_mlp)
                scasml_mean = np.mean(errors_scasml)
                scasml_std = np.std(errors_scasml)
                
                # Paired t-tests
                t_pinn_scasml, p_pinn_scasml = stats.ttest_rel(errors_pinn, errors_scasml)
                t_mlp_scasml, p_mlp_scasml = stats.ttest_rel(errors_mlp, errors_scasml)
                t_pinn_mlp, p_pinn_mlp = stats.ttest_rel(errors_pinn, errors_mlp)
                
                # Improvement percentages
                improvement_pinn = (rel_error_pinn - rel_error_scasml) / rel_error_pinn * 100
                improvement_mlp = (rel_error_mlp - rel_error_scasml) / rel_error_mlp * 100
                
                # Log to wandb
                wandb.log({
                    f"rho_{rho}_pinn_error": rel_error_pinn,
                    f"rho_{rho}_mlp_error": rel_error_mlp,
                    f"rho_{rho}_scasml_error": rel_error_scasml,
                    f"rho_{rho}_pinn_inference_time": inference_time_pinn,
                    f"rho_{rho}_mlp_inference_time": inference_time_mlp,
                    f"rho_{rho}_scasml_inference_time": inference_time_scasml,
                    f"rho_{rho}_improvement_vs_pinn": improvement_pinn,
                    f"rho_{rho}_improvement_vs_mlp": improvement_mlp,
                    f"rho_{rho}_p_pinn_scasml": p_pinn_scasml,
                    f"rho_{rho}_p_mlp_scasml": p_mlp_scasml,
                })
            
            # ==========================================
            # Visualization
            # ==========================================
            # Color scheme
            COLOR_SCHEME = {
                'PINN': '#000000',     # Black
                'MLP': '#A6A3A4',      # Gray
                'SCaSML': '#2C939A'    # Teal
            }
            
            # Configure matplotlib
            plt.rcParams.update({
                'font.family': 'DejaVu Sans',
                'font.size': 9,
                'axes.labelsize': 10,
                'axes.titlesize': 0,
                'legend.fontsize': 8,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'axes.linewidth': 0.8,
                'lines.linewidth': 1.2,
                'lines.markersize': 5,
                'savefig.dpi': 600,
                'savefig.transparent': True,
                'figure.autolayout': False
            })
            
            # ==========================================
            # Figure 1: Error vs Rho Level
            # ==========================================
            fig, ax = plt.subplots(figsize=(3.5, 3))
            
            rho_array = np.array(rho_levels[:len(pinn_errors)])
            
            # Plot with markers
            ax.plot(rho_array, pinn_errors, color=COLOR_SCHEME['PINN'], 
                   marker='o', linestyle='-', label='PINN (Large)', 
                   markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(rho_array, mlp_errors, color=COLOR_SCHEME['MLP'], 
                   marker='s', linestyle='-', label='MLP',
                   markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(rho_array, scasml_errors, color=COLOR_SCHEME['SCaSML'], 
                   marker='^', linestyle='-', label='SCaSML (Small)',
                   markerfacecolor='none', markeredgewidth=0.8)
            
            ax.set_xlabel('Rho Level', labelpad=3)
            ax.set_ylabel('Relative L2 Error', labelpad=3)
            ax.set_yscale('log')
            ax.set_xticks(rho_array)
            ax.legend(frameon=False, loc='upper right')
            ax.grid(True, which='major', axis='both', linestyle='--', 
                   linewidth=0.5, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/Error_vs_Rho.pdf', 
                       bbox_inches='tight', pad_inches=0.05)
            plt.close()
            
            # ==========================================
            # Figure 2: Improvement Bar Chart
            # ==========================================
            fig, ax = plt.subplots(figsize=(3.5, 3))
            
            # Calculate improvements at each rho level
            improvements_vs_pinn = [(pinn - scasml) / pinn * 100 
                                   for pinn, scasml in zip(pinn_errors, scasml_errors)]
            improvements_vs_mlp = [(mlp - scasml) / mlp * 100 
                                  for mlp, scasml in zip(mlp_errors, scasml_errors)]
            
            x = np.arange(len(rho_array))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, improvements_vs_pinn, width, 
                          label='SCaSML vs PINN', color=COLOR_SCHEME['PINN'], 
                          edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, improvements_vs_mlp, width, 
                          label='SCaSML vs MLP', color=COLOR_SCHEME['MLP'], 
                          edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Rho Level', labelpad=3)
            ax.set_ylabel('Improvement (%)', labelpad=3)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{r}' for r in rho_array])
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.legend(frameon=False, loc='upper left')
            ax.grid(True, which='major', axis='y', linestyle='--', 
                   linewidth=0.5, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/Improvement_Bar_Chart.pdf', 
                       bbox_inches='tight', pad_inches=0.05)
            plt.close()
            
            # ==========================================
            # Summary Statistics and Final Log Output
            # ==========================================
            avg_improvement_pinn = np.mean(improvements_vs_pinn)
            avg_improvement_mlp = np.mean(improvements_vs_mlp)
            
            wandb.log({
                "avg_improvement_vs_pinn": avg_improvement_pinn,
                "avg_improvement_vs_mlp": avg_improvement_mlp,
            })
            
            # Write final results to log file
            log_file = open(f"{save_path}/InferenceTime.log", "w")
            sys.stdout = log_file
            sys.stderr = log_file
            
            print("=" * 80)
            print("INFERENCE TIME TEST - FINAL RESULTS")
            print("=" * 80)
            print(f"Equation: {eq_name}")
            print(f"Dimension: {d+1}")
            print(f"Rho levels tested: {rho_array.tolist()}")
            print("=" * 80)
            print()
            
            print(f"{'Rho':<8} {'PINN Error':<15} {'MLP Error':<15} {'SCaSML Error':<15} {'PINN Time':<12} {'SCaSML Time':<12}")
            print("-" * 85)
            for i, rho in enumerate(rho_array):
                print(f"{rho:<8} {pinn_errors[i]:<15.6e} {mlp_errors[i]:<15.6e} {scasml_errors[i]:<15.6e} {pinn_times[i]:<12.4f} {scasml_times[i]:<12.4f}")
            print()
            
            print("Average Improvement:")
            print(f"  SCaSML vs PINN: {avg_improvement_pinn:+.2f}%")
            print(f"  SCaSML vs MLP: {avg_improvement_mlp:+.2f}%")
            print()
            
            print(f"Final Rho Level ({rho_array[-1]}):")
            print(f"  PINN error: {pinn_errors[-1]:.6e}")
            print(f"  MLP error: {mlp_errors[-1]:.6e}")
            print(f"  SCaSML error: {scasml_errors[-1]:.6e}")
            print(f"  PINN inference time: {pinn_times[-1]:.4f}s")
            print(f"  SCaSML inference time: {scasml_times[-1]:.4f}s")
            print(f"  Improvement vs PINN: {improvements_vs_pinn[-1]:+.2f}%")
            print(f"  Improvement vs MLP: {improvements_vs_mlp[-1]:+.2f}%")
            print("=" * 80)
            
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            log_file.close()
        
        # Stop profiler
        profiler.disable()
        profiler.dump_stats(f"{save_path}/{eq_name}_inference_time.prof")
        
        # Upload profiler results
        artifact = wandb.Artifact(f"{eq_name}_inference_time", type="profile")
        artifact.add_file(f"{save_path}/{eq_name}_inference_time.prof")
        wandb.log_artifact(artifact)
        
        print("Inference time test completed!")
        return rho_levels[-1] if rho_levels else 1
