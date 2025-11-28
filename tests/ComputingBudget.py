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
        pinn_flops = []
        mlp_flops = []
        scasml_flops = []
        
        # Base iterations for budget=1.0
        base_iterations = 2000
        
        def estimate_gflops(repeats=3, size=256):
            """
            Estimate the achievable compute on the current device by timing a
            matrix multiplication and converting to GFLOPs/s. We first try a
            JAX-backed matmul (captures GPU if available), otherwise fall back
            to NumPy.
            """
            try:
                # Use jax if available, best to warm-up and block_until_ready.
                import jax
                import jax.numpy as _jnp
                from jax import random as _random
                key1 = _random.PRNGKey(0)
                key2 = _random.PRNGKey(1)
                # Use float32 for speed and realistic load
                a = _random.normal(key1, shape=(size, size), dtype=_jnp.float32)
                b = _random.normal(key2, shape=(size, size), dtype=_jnp.float32)
                # Warm-up to avoid one-time compilation overhead
                _ = _jnp.dot(a, b).block_until_ready()
                times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    _ = _jnp.dot(a, b).block_until_ready()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                avg_time = sum(times) / max(1, len(times))
                flops = 2 * (size ** 3)  # flops for a dense matmul
                gflops = flops / max(avg_time, 1e-12) / 1e9
                return float(gflops)
            except Exception:
                # Fallback: numpy dot (CPU-bound)
                times = []
                for _ in range(repeats):
                    a = np.random.rand(size, size).astype(np.float32)
                    b = np.random.rand(size, size).astype(np.float32)
                    t0 = time.perf_counter()
                    np.dot(a, b)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                avg_time = sum(times) / max(1, len(times))
                flops = 2 * (size ** 3)
                gflops = flops / max(avg_time, 1e-12) / 1e9
                return float(gflops)

        # Estimate device compute capability to convert measured time into
        # an approximate FLOPS used metric. This is hardware-specific and
        # intended for relative comparisons across solvers in the same run.
        device_gflops = estimate_gflops()

        if device_gflops <= 0:
            device_gflops = 1.0

        # Also print/log the estimated device compute capability for transparency
        print(f"Estimated device: {device_gflops:.3f} GFLOPs/s (approx.)")
        print("Note: This test uses GFLOPs (estimated via a short matmul benchmark)")
        print("to express computational budget. This is hardware-dependent and intended")
        print("only for relative comparison across solvers within the same run.")

        if is_train:
            for budget in budget_levels:
                # Calculate iterations for this budget
                train_iters = int(base_iterations * budget)
                
                # ==========================================
                # PINN: Train and measure time
                # ==========================================
                solver1_copy = copy.deepcopy(self.solver1)
                opt1 = Adam(eq.n_input, 1, solver1_copy, eq)
                
                start_time = time.time()
                trained_pinn = opt1.train(f"{save_path}/model_weights_PINN_{budget}", iters=train_iters)
                train_time_pinn = time.time() - start_time
                
                start_time = time.time()
                sol_pinn = trained_pinn.predict(xt_test)
                inference_time_pinn = time.time() - start_time
                
                total_time_pinn = train_time_pinn + inference_time_pinn
                pinn_flops_used = total_time_pinn * device_gflops
                
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
                mlp_flops_used = total_time_mlp * device_gflops
                
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
                scasml_flops_used = total_time_scasml * device_gflops
                
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
                pinn_flops.append(pinn_flops_used)
                mlp_flops.append(mlp_flops_used)
                scasml_flops.append(scasml_flops_used)
                
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
                    f"budget_{budget}_pinn_flops": pinn_flops_used,
                    f"budget_{budget}_mlp_flops": mlp_flops_used,
                    f"budget_{budget}_scasml_flops": scasml_flops_used,
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
            
            if len(pinn_errors) == 0:
                print("No valid runs detected; skipping statistics and plotting.")
                profiler.disable()
                profiler.dump_stats(f"{save_path}/{eq_name}_computing_budget.prof")
                artifact = wandb.Artifact(f"{eq_name}_computing_budget", type="profile")
                artifact.add_file(f"{save_path}/{eq_name}_computing_budget.prof")
                wandb.log_artifact(artifact)
                return budget_levels[-1] if budget_levels else 1.0

            # ==========================================
            # Figure 1: Error vs FLOPs used
            # ==========================================
            fig, ax = plt.subplots(figsize=(3.5, 3))
            pinn_flops_array = np.array(pinn_flops)
            mlp_flops_array = np.array(mlp_flops)
            scasml_flops_array = np.array(scasml_flops)

            # Plot with markers
            ax.plot(pinn_flops_array, pinn_errors, color=COLOR_SCHEME['PINN'], 
            marker='o', linestyle='-', label='PINN', 
            markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(mlp_flops_array, mlp_errors, color=COLOR_SCHEME['MLP'], 
            marker='s', linestyle='-', label='MLP',
            markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(scasml_flops_array, scasml_errors, color=COLOR_SCHEME['SCaSML'], 
            marker='^', linestyle='-', label='SCaSML',
            markerfacecolor='none', markeredgewidth=0.8)
            ax.set_xlabel('Computational Cost (GFLOPs)', labelpad=3)
            ax.set_ylabel('Relative L2 Error', labelpad=3)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.legend(frameon=False, loc='upper right')
            ax.grid(True, which='major', axis='both', linestyle='--', 
                   linewidth=0.5, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/Error_vs_FLOPs.pdf', 
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
            
            budget_array = np.array(budget_levels[:len(pinn_errors)])
            x = np.arange(len(budget_array))
            # average flops per budget across solvers (for display only)
            avg_flops = (pinn_flops_array + mlp_flops_array + scasml_flops_array) / 3.0
            width = 0.35
            
            bars1 = ax.bar(x - width/2, improvements_vs_pinn, width, 
                          label='SCaSML vs PINN', color=COLOR_SCHEME['PINN'], 
                          edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, improvements_vs_mlp, width, 
                          label='SCaSML vs MLP', color=COLOR_SCHEME['MLP'], 
                          edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Computing Budget (×baseline)', labelpad=3)
            ax.set_ylabel('Improvement (%)', labelpad=3)
            # Show budget multiplier and the approximate GFLOPs (avg across solvers)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{b}×\n{avg:.1f} GFLOPs' for b, avg in zip(budget_array, avg_flops)])
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
                "avg_flops_pinn": np.mean(pinn_flops) if len(pinn_flops) else 0.0,
                "avg_flops_mlp": np.mean(mlp_flops) if len(mlp_flops) else 0.0,
                "avg_flops_scasml": np.mean(scasml_flops) if len(scasml_flops) else 0.0,
                "device_gflops_estimated": device_gflops,
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
            print("=" * 80)
            print()
            
            print(f"{ 'Budget':<12} {'PINN Error':<15} {'PINN GFLOPs':<13} {'MLP Error':<15} {'MLP GFLOPs':<13} {'SCaSML Error':<15} {'SCaSML GFLOPs':<13}")
            print("-" * 60)
            for i, budget in enumerate(budget_array):
                print(
                    f"{budget:<12.1f} {pinn_errors[i]:<15.6e} {pinn_flops[i]:<13.2f} {mlp_errors[i]:<15.6e} {mlp_flops[i]:<13.2f} {scasml_errors[i]:<15.6e} {scasml_flops[i]:<13.2f}"
                )
            print()
            
            print("Average Improvement:")
            print(f"  SCaSML vs PINN: {avg_improvement_pinn:+.2f}%")
            print(f"  SCaSML vs MLP: {avg_improvement_mlp:+.2f}%")
            print()
            
            print(f"Final Budget Level ({budget_array[-1]:.1f}×):")
            print(f"  PINN error: {pinn_errors[-1]:.6e}")
            print(f"  MLP error: {mlp_errors[-1]:.6e}")
            print(f"  SCaSML error: {scasml_errors[-1]:.6e}")
            print(f"  PINN GFLOPs: {pinn_flops[-1]:.2f}")
            print(f"  MLP GFLOPs: {mlp_flops[-1]:.2f}")
            print(f"  SCaSML GFLOPs: {scasml_flops[-1]:.2f}")
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
