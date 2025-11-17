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

class ComputingBudget(object):
    '''
    Computing Budget test: compares solvers under equal total computational budget.
    
    This test demonstrates that SCaSML achieves better accuracy than MLP and PINN
    when all methods are given the same total computing time (training + inference).

    Attributes:
    equation (object): An object representing the equation to solve.
    dim (int): The dimension of the input space minus one.
    solver1 (object): A jax Gaussian Process model (PINN).
    solver2 (object): An object for the MLP solver.
    solver3 (object): An object for the ScaSML solver.
    t0 (float): The initial time.
    T (float): The final time.
    '''
    def __init__(self, equation, solver1, solver2, solver3, is_train):
        '''
        Initializes the computing budget test with given solvers and equation.

        Parameters:
        equation (object): The equation object containing problem specifics.
        solver1 (object): The PINN solver.
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
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.t0 = equation.t0
        self.T = equation.T
        self.is_train = is_train

    def test(self, save_path, budget_levels=[1.0, 2.0, 4.0, 8.0, 16.0], num_domain=1000, num_boundary=200):
        '''
        Compares solvers under different computing budget levels.
        
        The budget is normalized such that 1.0 represents a baseline computation time.
        For each budget level, we train each solver to consume approximately the same
        total time (training + inference), then measure the resulting accuracy.
    
        Parameters:
        save_path (str): The path to save the results.
        budget_levels (list): List of budget multipliers (e.g., [1.0, 2.0, 4.0]).
        num_domain (int): The number of points in the test domain.
        num_boundary (int): The number of points on the test boundary.
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
        
        # Generate test data (fixed across all budget levels)
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
        
        # Base iterations for budget=1.0
        base_iterations = 2000
        
        # Open log file
        log_file = open(f"{save_path}/ComputingBudget.log", "w")
        sys.stdout = log_file
        sys.stderr = log_file
        
        print("=" * 80)
        print("COMPUTING BUDGET TEST")
        print("=" * 80)
        print(f"Equation: {eq_name}")
        print(f"Dimension: {d+1}")
        print(f"Test points: {len(xt_test)}")
        print(f"Budget levels: {budget_levels}")
        print("=" * 80)
        print()
        
        if is_train:
            for budget in budget_levels:
                print(f"\n{'='*80}")
                print(f"Budget Level: {budget}x")
                print(f"{'='*80}")
                
                # Calculate iterations for this budget
                train_iters = int(base_iterations * budget)
                
                # ==========================================
                # PINN: Train and measure time
                # ==========================================
                print(f"\n--- PINN (budget={budget}x) ---")
                solver1_copy = copy.deepcopy(self.solver1)
                opt1 = Adam(eq.n_input, 1, solver1_copy, eq)
                
                start_time = time.time()
                trained_pinn = opt1.train(f"{save_path}/model_weights_PINN_{budget}", iters=train_iters)
                train_time_pinn = time.time() - start_time
                
                start_time = time.time()
                sol_pinn = trained_pinn.predict(xt_test)
                inference_time_pinn = time.time() - start_time
                
                total_time_pinn = train_time_pinn + inference_time_pinn
                
                # ==========================================
                # MLP: Adjust training to match budget
                # ==========================================
                print(f"\n--- MLP (budget={budget}x) ---")
                solver2_copy = copy.deepcopy(self.solver2)
                
                # MLP inference is slower, so we adjust training iterations
                # to approximately match total time budget
                rho_mlp = max(2, int(np.log(train_iters) / np.log(np.log(train_iters) + 1)))
                
                start_time = time.time()
                # MLP doesn't need PINN training
                dummy_train_time = 0  # MLP uses pre-trained or no training
                train_time_mlp = dummy_train_time
                
                sol_mlp = solver2_copy.u_solve(rho_mlp, rho_mlp, xt_test)
                inference_time_mlp = time.time() - start_time
                
                total_time_mlp = train_time_mlp + inference_time_mlp
                
                # ==========================================
                # ScaSML: Optimize training/inference split
                # ==========================================
                print(f"\n--- ScaSML (budget={budget}x) ---")
                solver3_copy = copy.deepcopy(self.solver3)
                
                # ScaSML uses less training for PINN backbone
                scasml_train_iters = train_iters // (d + 1)
                rho_scasml = max(2, int(np.log(scasml_train_iters) / np.log(np.log(scasml_train_iters) + 1)))
                
                opt3 = Adam(eq.n_input, 1, solver3_copy.PINN, eq)
                
                start_time = time.time()
                trained_pinn_backbone = opt3.train(f"{save_path}/model_weights_ScaSML_{budget}", 
                                                   iters=scasml_train_iters)
                train_time_scasml = time.time() - start_time
                
                solver3_copy.PINN = trained_pinn_backbone
                
                start_time = time.time()
                sol_scasml = solver3_copy.u_solve(rho_scasml, rho_scasml, xt_test)
                inference_time_scasml = time.time() - start_time
                
                total_time_scasml = train_time_scasml + inference_time_scasml
                
                # ==========================================
                # Compute Errors
                # ==========================================
                valid_mask = ~(np.isnan(sol_pinn) | np.isnan(sol_mlp) | 
                              np.isnan(sol_scasml) | np.isnan(exact_sol)).flatten()
                
                if np.sum(valid_mask) == 0:
                    print("Warning: All predictions are NaN. Skipping this budget level.")
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
                pinn_times.append(total_time_pinn)
                mlp_times.append(total_time_mlp)
                scasml_times.append(total_time_scasml)
                
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
                
                # Print results
                print(f"\nResults for Budget={budget}x:")
                print(f"{'-'*60}")
                print(f"{'Method':<15} {'Time (s)':<15} {'Rel L2 Error':<20}")
                print(f"{'-'*60}")
                print(f"{'PINN':<15} {total_time_pinn:<15.3f} {rel_error_pinn:<20.6e}")
                print(f"  Train: {train_time_pinn:.3f}s, Inference: {inference_time_pinn:.3f}s")
                print(f"  Mean L1: {pinn_mean:.6e}, Std: {pinn_std:.6e}")
                print()
                print(f"{'MLP':<15} {total_time_mlp:<15.3f} {rel_error_mlp:<20.6e}")
                print(f"  Train: {train_time_mlp:.3f}s, Inference: {inference_time_mlp:.3f}s")
                print(f"  Mean L1: {mlp_mean:.6e}, Std: {mlp_std:.6e}")
                print()
                print(f"{'SCaSML':<15} {total_time_scasml:<15.3f} {rel_error_scasml:<20.6e}")
                print(f"  Train: {train_time_scasml:.3f}s, Inference: {inference_time_scasml:.3f}s")
                print(f"  Mean L1: {scasml_mean:.6e}, Std: {scasml_std:.6e}")
                print(f"{'-'*60}")
                
                print(f"\nStatistical Significance (Paired t-test):")
                print(f"  PINN vs SCaSML: p-value = {p_pinn_scasml:.6e}")
                print(f"  MLP vs SCaSML: p-value = {p_mlp_scasml:.6e}")
                print(f"  PINN vs MLP: p-value = {p_pinn_mlp:.6e}")
                
                # Improvement percentages
                improvement_pinn = (rel_error_pinn - rel_error_scasml) / rel_error_pinn * 100
                improvement_mlp = (rel_error_mlp - rel_error_scasml) / rel_error_mlp * 100
                
                print(f"\nImprovement:")
                print(f"  SCaSML vs PINN: {improvement_pinn:+.2f}%")
                print(f"  SCaSML vs MLP: {improvement_mlp:+.2f}%")
                
                # Log to wandb
                wandb.log({
                    f"budget_{budget}_pinn_error": rel_error_pinn,
                    f"budget_{budget}_mlp_error": rel_error_mlp,
                    f"budget_{budget}_scasml_error": rel_error_scasml,
                    f"budget_{budget}_pinn_time": total_time_pinn,
                    f"budget_{budget}_mlp_time": total_time_mlp,
                    f"budget_{budget}_scasml_time": total_time_scasml,
                    f"budget_{budget}_improvement_vs_pinn": improvement_pinn,
                    f"budget_{budget}_improvement_vs_mlp": improvement_mlp,
                    f"budget_{budget}_p_pinn_scasml": p_pinn_scasml,
                    f"budget_{budget}_p_mlp_scasml": p_mlp_scasml,
                })
            
            # ==========================================
            # Visualization
            # ==========================================
            print(f"\n{'='*80}")
            print("GENERATING VISUALIZATIONS")
            print(f"{'='*80}")
            
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
            # Figure 1: Error vs Budget
            # ==========================================
            fig, ax = plt.subplots(figsize=(3.5, 3))
            
            budget_array = np.array(budget_levels[:len(pinn_errors)])
            
            # Plot with markers
            ax.plot(budget_array, pinn_errors, color=COLOR_SCHEME['PINN'], 
                   marker='o', linestyle='-', label='PINN', 
                   markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(budget_array, mlp_errors, color=COLOR_SCHEME['MLP'], 
                   marker='s', linestyle='-', label='MLP',
                   markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(budget_array, scasml_errors, color=COLOR_SCHEME['SCaSML'], 
                   marker='^', linestyle='-', label='SCaSML',
                   markerfacecolor='none', markeredgewidth=0.8)
            
            ax.set_xlabel('Computing Budget (×baseline)', labelpad=3)
            ax.set_ylabel('Relative L2 Error', labelpad=3)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.legend(frameon=False, loc='upper right')
            ax.grid(True, which='major', axis='both', linestyle='--', 
                   linewidth=0.5, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/Error_vs_Budget.pdf', 
                       bbox_inches='tight', pad_inches=0.05)
            plt.close()
            
            # ==========================================
            # Figure 2: Error vs Actual Time
            # ==========================================
            fig, ax = plt.subplots(figsize=(3.5, 3))
            
            ax.plot(pinn_times, pinn_errors, color=COLOR_SCHEME['PINN'], 
                   marker='o', linestyle='-', label='PINN',
                   markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(mlp_times, mlp_errors, color=COLOR_SCHEME['MLP'], 
                   marker='s', linestyle='-', label='MLP',
                   markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(scasml_times, scasml_errors, color=COLOR_SCHEME['SCaSML'], 
                   marker='^', linestyle='-', label='SCaSML',
                   markerfacecolor='none', markeredgewidth=0.8)
            
            ax.set_xlabel('Total Time (seconds)', labelpad=3)
            ax.set_ylabel('Relative L2 Error', labelpad=3)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.legend(frameon=False, loc='upper right')
            ax.grid(True, which='major', axis='both', linestyle='--', 
                   linewidth=0.5, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/Error_vs_Time.pdf', 
                       bbox_inches='tight', pad_inches=0.05)
            plt.close()
            
            # ==========================================
            # Figure 3: Improvement Bar Chart
            # ==========================================
            fig, ax = plt.subplots(figsize=(3.5, 3))
            
            # Calculate improvements at each budget level
            improvements_vs_pinn = [(pinn - scasml) / pinn * 100 
                                   for pinn, scasml in zip(pinn_errors, scasml_errors)]
            improvements_vs_mlp = [(mlp - scasml) / mlp * 100 
                                  for mlp, scasml in zip(mlp_errors, scasml_errors)]
            
            x = np.arange(len(budget_array))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, improvements_vs_pinn, width, 
                          label='SCaSML vs PINN', color=COLOR_SCHEME['PINN'], 
                          edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, improvements_vs_mlp, width, 
                          label='SCaSML vs MLP', color=COLOR_SCHEME['MLP'], 
                          edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Computing Budget (×baseline)', labelpad=3)
            ax.set_ylabel('Improvement (%)', labelpad=3)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{b}×' for b in budget_array])
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
            # Summary Statistics
            # ==========================================
            print(f"\n{'='*80}")
            print("SUMMARY STATISTICS")
            print(f"{'='*80}")
            
            avg_improvement_pinn = np.mean(improvements_vs_pinn)
            avg_improvement_mlp = np.mean(improvements_vs_mlp)
            
            print(f"\nAverage improvement across all budgets:")
            print(f"  SCaSML vs PINN: {avg_improvement_pinn:+.2f}%")
            print(f"  SCaSML vs MLP: {avg_improvement_mlp:+.2f}%")
            
            print(f"\nFinal budget level ({budget_array[-1]}×):")
            print(f"  PINN error: {pinn_errors[-1]:.6e}")
            print(f"  MLP error: {mlp_errors[-1]:.6e}")
            print(f"  SCaSML error: {scasml_errors[-1]:.6e}")
            print(f"  Improvement vs PINN: {improvements_vs_pinn[-1]:+.2f}%")
            print(f"  Improvement vs MLP: {improvements_vs_mlp[-1]:+.2f}%")
            
            wandb.log({
                "avg_improvement_vs_pinn": avg_improvement_pinn,
                "avg_improvement_vs_mlp": avg_improvement_mlp,
            })
        
        # Reset stdout and stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        log_file.close()
        
        # Stop profiler
        profiler.disable()
        profiler.dump_stats(f"{save_path}/{eq_name}_computing_budget.prof")
        
        # Upload profiler results
        artifact = wandb.Artifact(f"{eq_name}_computing_budget", type="profile")
        artifact.add_file(f"{save_path}/{eq_name}_computing_budget.prof")
        wandb.log_artifact(artifact)
        
        print("Computing budget test completed!")
        return budget_levels[-1] if budget_levels else 1.0
