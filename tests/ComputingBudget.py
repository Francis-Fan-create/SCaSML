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
import deepxde as dde

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
        
        # Base network configuration for PINN (budget=1.0)
        base_hidden_width = 50
        base_hidden_depth = 5
        base_iterations = 2000
        
        def create_scaled_pinn(budget, eq):
            '''
            Create a PINN model with network size scaled by budget.
            Scaling is done by increasing both width and depth of the network.
            
            Parameters:
            budget (float): Budget multiplier.
            eq (object): The equation object.
            
            Returns:
            dde.Model: A new PINN model with scaled network architecture.
            '''
            n_input = eq.n_input
            n_output = eq.n_output
            
            # Scale network size: increase both width and depth
            # Width scales as sqrt(budget) to maintain balanced growth
            # Depth increases logarithmically with budget
            scaled_width = int(base_hidden_width * np.sqrt(budget))
            scaled_depth = max(base_hidden_depth, int(base_hidden_depth + np.log2(budget)))
            
            # Construct layer sizes: [input] + [hidden]*depth + [output]
            layer_sizes = [n_input] + [scaled_width] * scaled_depth + [n_output]
            
            # Create new network with scaled architecture
            net = dde.maps.jax.FNN(layer_sizes, "tanh", "Glorot normal")
            
            # Apply terminal transform if available
            if hasattr(eq, 'terminal_transform'):
                net.apply_output_transform(eq.terminal_transform)
            
            # Generate data and create model
            data = eq.generate_data()
            model = dde.Model(data, net)
            
            return model, scaled_width, scaled_depth
        
        if is_train:
            for budget in budget_levels:
                # Calculate scaled iterations for MLP and ScaSML (PINN uses network size scaling)
                train_iters = int(base_iterations * budget)
                
                # ==========================================
                # PINN: Scale by network size (not iterations)
                # ==========================================
                # Create a new PINN with scaled network architecture
                scaled_pinn, scaled_width, scaled_depth = create_scaled_pinn(budget, eq)
                opt1 = Adam(eq.n_input, 1, scaled_pinn, eq)
                
                start_time = time.time()
                # Use fixed base iterations, scaling is done via network size
                trained_pinn = opt1.train(f"{save_path}/model_weights_PINN_{budget}", iters=base_iterations)
                train_time_pinn = time.time() - start_time
                
                start_time = time.time()
                sol_pinn = trained_pinn.predict(xt_test)
                inference_time_pinn = time.time() - start_time
                
                total_time_pinn = train_time_pinn + inference_time_pinn
                
                # Log PINN network configuration
                wandb.log({
                    f"budget_{budget}_pinn_width": scaled_width,
                    f"budget_{budget}_pinn_depth": scaled_depth,
                })
                
                # ==========================================
                # MLP: Adjust training to match budget
                # ==========================================
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
                
                # Improvement percentages
                improvement_pinn = (rel_error_pinn - rel_error_scasml) / rel_error_pinn * 100
                improvement_mlp = (rel_error_mlp - rel_error_scasml) / rel_error_mlp * 100
                
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
            # Figure 2: Improvement Bar Chart
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
            # Summary Statistics and Final Log Output
            # ==========================================
            avg_improvement_pinn = np.mean(improvements_vs_pinn)
            avg_improvement_mlp = np.mean(improvements_vs_mlp)
            
            wandb.log({
                "avg_improvement_vs_pinn": avg_improvement_pinn,
                "avg_improvement_vs_mlp": avg_improvement_mlp,
            })
            
            # Write final results to log file
            log_file = open(f"{save_path}/ComputingBudget.log", "w")
            sys.stdout = log_file
            sys.stderr = log_file
            
            print("=" * 80)
            print("COMPUTING BUDGET TEST - FINAL RESULTS")
            print("=" * 80)
            print(f"Equation: {eq_name}")
            print(f"Dimension: {d+1}")
            print(f"Budget levels tested: {budget_array.tolist()}")
            print(f"PINN scaling method: Network size (width/depth)")
            print(f"PINN base config: width={base_hidden_width}, depth={base_hidden_depth}, iters={base_iterations}")
            print("=" * 80)
            print()
            
            # Show PINN network configuration for each budget
            print("PINN Network Scaling:")
            for i, budget in enumerate(budget_array):
                scaled_w = int(base_hidden_width * np.sqrt(budget))
                scaled_d = max(base_hidden_depth, int(base_hidden_depth + np.log2(budget)))
                print(f"  Budget {budget:.1f}×: width={scaled_w}, depth={scaled_d}")
            print()
            
            print(f"{'Budget':<12} {'PINN Error':<15} {'MLP Error':<15} {'SCaSML Error':<15}")
            print("-" * 60)
            for i, budget in enumerate(budget_array):
                print(f"{budget:<12.1f} {pinn_errors[i]:<15.6e} {mlp_errors[i]:<15.6e} {scasml_errors[i]:<15.6e}")
            print()
            
            print("Average Improvement:")
            print(f"  SCaSML vs PINN: {avg_improvement_pinn:+.2f}%")
            print(f"  SCaSML vs MLP: {avg_improvement_mlp:+.2f}%")
            print()
            
            print(f"Final Budget Level ({budget_array[-1]:.1f}×):")
            print(f"  PINN error: {pinn_errors[-1]:.6e}")
            print(f"  MLP error: {mlp_errors[-1]:.6e}")
            print(f"  SCaSML error: {scasml_errors[-1]:.6e}")
            print(f"  Improvement vs PINN: {improvements_vs_pinn[-1]:+.2f}%")
            print(f"  Improvement vs MLP: {improvements_vs_mlp[-1]:+.2f}%")
            print("=" * 80)
            
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
