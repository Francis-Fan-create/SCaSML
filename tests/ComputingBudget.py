import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import time
import sys
import os
import gc
import cProfile
import shutil
import copy
from optimizers.Adam import Adam
import jax.numpy as jnp
import jax
from scipy import stats
import matplotlib.ticker as ticker
import deepxde as dde
from solvers.MLP import MLP
from solvers.ScaSML import ScaSML


def clear_gpu_memory():
    """Clear GPU memory for both PyTorch and JAX."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # Clear JAX caches
    try:
        jax.clear_caches()
    except:
        pass

class ComputingBudget(object):
    '''
    Computing Budget test: compares solvers under equal total computational budget (GFlops).
    
    This test demonstrates that SCaSML achieves better accuracy than MLP and PINN
    when all methods are given the same total computational budget measured in GFlops.
    The budget is controlled by varying network architecture (layers and neurons).

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

    def compute_fnn_gflops(self, layer_sizes, batch_size=1, include_activation=True):
        '''
        Compute the GFlops for a feedforward neural network (FNN) inference.
        
        For each fully connected layer: FLOPs = 2 * input_dim * output_dim (multiply-add)
        Activation functions add approximately output_dim FLOPs per layer.
        
        Parameters:
        layer_sizes (list): List of layer sizes, e.g., [21, 50, 50, 50, 1]
        batch_size (int): Batch size for inference
        include_activation (bool): Whether to include activation function FLOPs
        
        Returns:
        float: GFlops (10^9 FLOPs)
        '''
        total_flops = 0
        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            # Matrix multiplication: 2 * input * output (multiply-add counted as 2 ops)
            total_flops += 2 * input_dim * output_dim * batch_size
            # Activation function (tanh approximation: ~10 FLOPs per element)
            if include_activation and i < len(layer_sizes) - 2:  # No activation on output layer
                total_flops += 10 * output_dim * batch_size
        return total_flops / 1e9  # Convert to GFlops

    def generate_network_configs(self, n_input, target_gflops_list):
        '''
        Generate network configurations (layer sizes) that achieve target GFlops.
        
        Parameters:
        n_input (int): Input dimension
        target_gflops_list (list): List of target GFlops values
        
        Returns:
        list: List of tuples (layer_sizes, actual_gflops)
        '''
        configs = []
        
        # Define a range of network architectures with increasing complexity
        # Format: (num_layers, neurons_per_layer)
        # REDUCED sizes for lower GPU memory usage
        architecture_options = [
            (2, 16),   # Tiny
            (2, 24),   # Small
            (3, 32),   # Medium-Small
            (3, 40),   # Medium (baseline)
            (4, 48),   # Medium-Large
            (4, 56),   # Large
        ]
        
        # Compute GFlops for each architecture and select closest to targets
        for target_gflops in target_gflops_list:
            best_config = None
            best_diff = float('inf')
            
            for num_layers, neurons in architecture_options:
                layer_sizes = [n_input] + [neurons] * num_layers + [1]
                gflops = self.compute_fnn_gflops(layer_sizes)
                diff = abs(gflops - target_gflops)
                
                if diff < best_diff:
                    best_diff = diff
                    best_config = (layer_sizes, gflops)
            
            # If no close match found, interpolate to create custom architecture
            if best_config is None or best_diff > target_gflops * 0.5:
                # Estimate neurons needed for target GFlops with 3 layers (reduced from 5)
                # Rough approximation: GFlops ≈ 2 * n_input * neurons + 2 * 2 * neurons^2
                neurons = int(np.sqrt(target_gflops * 1e9 / (2 * 3)))
                neurons = max(16, min(neurons, 64))  # Clamp to smaller range for memory
                layer_sizes = [n_input] + [neurons] * 3 + [1]
                gflops = self.compute_fnn_gflops(layer_sizes)
                best_config = (layer_sizes, gflops)
            
            configs.append(best_config)
        
        return configs

    def create_pinn_model(self, layer_sizes, equation):
        '''
        Create a new PINN model with the specified layer sizes.
        
        Parameters:
        layer_sizes (list): Network architecture, e.g., [21, 50, 50, 1]
        equation (object): The equation object
        
        Returns:
        dde.Model: A new PINN model
        '''
        net = dde.maps.jax.FNN(layer_sizes, "tanh", "Glorot normal")
        terminal_transform = equation.terminal_transform
        net.apply_output_transform(terminal_transform)
        data = equation.generate_data()
        model = dde.Model(data, net)
        return model

    def test(self, save_path, gflops_levels=None, num_domain=200, num_boundary=50, train_iters=500):
        '''
        Compares solvers under different computing budget levels measured in GFlops.
        
        The budget is controlled by varying network architecture (layers and neurons).
        For each GFlops level, we train networks of corresponding size, then measure 
        the resulting accuracy.
    
        Parameters:
        save_path (str): The path to save the results.
        gflops_levels (list): List of target GFlops values. If None, uses default values.
        num_domain (int): The number of points in the test domain (reduced default: 200).
        num_boundary (int): The number of points on the test boundary (reduced default: 50).
        train_iters (int): Number of training iterations for each network (reduced default: 500).
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
        n_input = eq.n_input
        
        # Default GFlops levels if not specified
        if gflops_levels is None:
            # Generate a smaller range of GFlops based on dimension
            # Reduced number of levels (4 instead of 6) for faster testing
            base_gflops = 1e-6 * n_input  # Scale with input dimension
            gflops_levels = [base_gflops * mult for mult in [1, 4, 16, 64]]
        
        # Generate network configurations for each GFlops level
        network_configs = self.generate_network_configs(n_input, gflops_levels)
        
        # Generate test data (fixed across all budget levels)
        data_domain_test, data_boundary_test = eq.generate_test_data(num_domain, num_boundary)
        xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
        exact_sol = eq.exact_solution(xt_test)
        
        # Storage for results
        pinn_errors = []
        mlp_errors = []
        scasml_errors = []
        actual_gflops_list = []
        pinn_times = []
        mlp_times = []
        scasml_times = []
        layer_sizes_list = []
        
        if is_train:
            for config_idx, (layer_sizes, actual_gflops) in enumerate(network_configs):
                print(f"\n{'='*60}")
                print(f"Testing network config {config_idx+1}/{len(network_configs)}")
                print(f"Layer sizes: {layer_sizes}")
                print(f"Target GFlops: {gflops_levels[config_idx]:.2e}, Actual GFlops: {actual_gflops:.2e}")
                print(f"{'='*60}")
                
                # Clear GPU memory before each configuration
                clear_gpu_memory()
                
                actual_gflops_list.append(actual_gflops)
                layer_sizes_list.append(layer_sizes)
                
                # ==========================================
                # PINN: Create and train with current architecture
                # ==========================================
                pinn_model = self.create_pinn_model(layer_sizes, eq)
                opt1 = Adam(eq.n_input, 1, pinn_model, eq)
                
                start_time = time.time()
                trained_pinn = opt1.train(f"{save_path}/model_weights_PINN_{config_idx}", 
                                         iters=train_iters)
                train_time_pinn = time.time() - start_time
                
                start_time = time.time()
                sol_pinn = trained_pinn.predict(xt_test)
                inference_time_pinn = time.time() - start_time
                
                total_time_pinn = train_time_pinn + inference_time_pinn
                
                # ==========================================
                # MLP: Use standard MLP (no neural network, just Picard iteration)
                # ==========================================
                # Clear memory before MLP
                clear_gpu_memory()
                
                solver2_copy = copy.deepcopy(self.solver2)
                
                # Adjust rho based on network complexity (reduced max from 6 to 4)
                rho_mlp = max(2, min(4, int(np.log(actual_gflops * 1e9 + 1) / 2)))
                
                start_time = time.time()
                sol_mlp = solver2_copy.u_solve(rho_mlp, rho_mlp, xt_test)
                inference_time_mlp = time.time() - start_time
                total_time_mlp = inference_time_mlp
                
                # ==========================================
                # ScaSML: Create PINN backbone with current architecture
                # ==========================================
                # Clear memory before ScaSML
                clear_gpu_memory()
                
                scasml_pinn_model = self.create_pinn_model(layer_sizes, eq)
                opt3 = Adam(eq.n_input, 1, scasml_pinn_model, eq)
                
                # ScaSML uses fewer training iterations for backbone
                scasml_train_iters = max(50, train_iters // (d + 1))
                rho_scasml = max(2, min(4, int(np.log(actual_gflops * 1e9 + 1) / 2)))
                
                start_time = time.time()
                trained_scasml_backbone = opt3.train(f"{save_path}/model_weights_ScaSML_{config_idx}", 
                                                     iters=scasml_train_iters)
                train_time_scasml = time.time() - start_time
                
                # Create ScaSML solver with trained backbone
                scasml_solver = ScaSML(equation=eq, PINN=trained_scasml_backbone)
                
                start_time = time.time()
                sol_scasml = scasml_solver.u_solve(rho_scasml, rho_scasml, xt_test)
                inference_time_scasml = time.time() - start_time
                
                total_time_scasml = train_time_scasml + inference_time_scasml
                
                # Clear memory after each solver
                clear_gpu_memory()
                del scasml_solver, trained_scasml_backbone, opt3, scasml_pinn_model
                clear_gpu_memory()
                
                # ==========================================
                # Compute Errors
                # ==========================================
                valid_mask = ~(np.isnan(sol_pinn) | np.isnan(sol_mlp) | 
                              np.isnan(sol_scasml) | np.isnan(exact_sol)).flatten()
                
                if np.sum(valid_mask) == 0:
                    print(f"Warning: No valid samples for config {config_idx}")
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
                # Paired t-tests
                t_pinn_scasml, p_pinn_scasml = stats.ttest_rel(errors_pinn, errors_scasml)
                t_mlp_scasml, p_mlp_scasml = stats.ttest_rel(errors_mlp, errors_scasml)
                
                # Improvement percentages
                improvement_pinn = (rel_error_pinn - rel_error_scasml) / rel_error_pinn * 100 if rel_error_pinn > 0 else 0
                improvement_mlp = (rel_error_mlp - rel_error_scasml) / rel_error_mlp * 100 if rel_error_mlp > 0 else 0
                
                # Log to wandb
                wandb.log({
                    f"gflops_{actual_gflops:.2e}_pinn_error": rel_error_pinn,
                    f"gflops_{actual_gflops:.2e}_mlp_error": rel_error_mlp,
                    f"gflops_{actual_gflops:.2e}_scasml_error": rel_error_scasml,
                    f"gflops_{actual_gflops:.2e}_pinn_time": total_time_pinn,
                    f"gflops_{actual_gflops:.2e}_mlp_time": total_time_mlp,
                    f"gflops_{actual_gflops:.2e}_scasml_time": total_time_scasml,
                    f"gflops_{actual_gflops:.2e}_improvement_vs_pinn": improvement_pinn,
                    f"gflops_{actual_gflops:.2e}_improvement_vs_mlp": improvement_mlp,
                    f"gflops_{actual_gflops:.2e}_p_pinn_scasml": p_pinn_scasml,
                    f"gflops_{actual_gflops:.2e}_p_mlp_scasml": p_mlp_scasml,
                    f"gflops_{actual_gflops:.2e}_layer_sizes": str(layer_sizes),
                })
                
                print(f"PINN error: {rel_error_pinn:.6e}, MLP error: {rel_error_mlp:.6e}, ScaSML error: {rel_error_scasml:.6e}")
                
                # Clean up after each iteration
                del trained_pinn, pinn_model, opt1, solver2_copy
                del sol_pinn, sol_mlp, sol_scasml
                clear_gpu_memory()
            
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
            # Figure 1: Error vs GFlops
            # ==========================================
            fig, ax = plt.subplots(figsize=(3.5, 3))
            
            gflops_array = np.array(actual_gflops_list[:len(pinn_errors)])
            
            # Plot with markers
            ax.plot(gflops_array, pinn_errors, color=COLOR_SCHEME['PINN'], 
                   marker='o', linestyle='-', label='PINN', 
                   markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(gflops_array, mlp_errors, color=COLOR_SCHEME['MLP'], 
                   marker='s', linestyle='-', label='MLP',
                   markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(gflops_array, scasml_errors, color=COLOR_SCHEME['SCaSML'], 
                   marker='^', linestyle='-', label='SCaSML',
                   markerfacecolor='none', markeredgewidth=0.8)
            
            ax.set_xlabel('GFlops', labelpad=3)
            ax.set_ylabel('Relative L2 Error', labelpad=3)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.legend(frameon=False, loc='upper right')
            ax.grid(True, which='major', axis='both', linestyle='--', 
                   linewidth=0.5, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Format x-axis with scientific notation
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/Error_vs_GFlops.pdf', 
                       bbox_inches='tight', pad_inches=0.05)
            plt.close()
            
            # ==========================================
            # Figure 2: Improvement Bar Chart
            # ==========================================
            fig, ax = plt.subplots(figsize=(3.5, 3))
            
            # Calculate improvements at each GFlops level
            improvements_vs_pinn = [(pinn - scasml) / pinn * 100 if pinn > 0 else 0
                                   for pinn, scasml in zip(pinn_errors, scasml_errors)]
            improvements_vs_mlp = [(mlp - scasml) / mlp * 100 if mlp > 0 else 0
                                  for mlp, scasml in zip(mlp_errors, scasml_errors)]
            
            x = np.arange(len(gflops_array))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, improvements_vs_pinn, width, 
                          label='SCaSML vs PINN', color=COLOR_SCHEME['PINN'], 
                          edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, improvements_vs_mlp, width, 
                          label='SCaSML vs MLP', color=COLOR_SCHEME['MLP'], 
                          edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('GFlops', labelpad=3)
            ax.set_ylabel('Improvement (%)', labelpad=3)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{g:.1e}' for g in gflops_array], rotation=45, ha='right')
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
            # Figure 3: Time vs GFlops
            # ==========================================
            fig, ax = plt.subplots(figsize=(3.5, 3))
            
            ax.plot(gflops_array, pinn_times, color=COLOR_SCHEME['PINN'], 
                   marker='o', linestyle='-', label='PINN', 
                   markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(gflops_array, mlp_times, color=COLOR_SCHEME['MLP'], 
                   marker='s', linestyle='-', label='MLP',
                   markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(gflops_array, scasml_times, color=COLOR_SCHEME['SCaSML'], 
                   marker='^', linestyle='-', label='SCaSML',
                   markerfacecolor='none', markeredgewidth=0.8)
            
            ax.set_xlabel('GFlops', labelpad=3)
            ax.set_ylabel('Total Time (s)', labelpad=3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(frameon=False, loc='upper left')
            ax.grid(True, which='major', axis='both', linestyle='--', 
                   linewidth=0.5, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/Time_vs_GFlops.pdf', 
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
            
            print("=" * 100)
            print("COMPUTING BUDGET TEST - FINAL RESULTS (GFlops-based)")
            print("=" * 100)
            print(f"Equation: {eq_name}")
            print(f"Dimension: {d+1}")
            print(f"Training iterations per network: {train_iters}")
            print("=" * 100)
            print()
            
            print(f"{'GFlops':<15} {'Layer Sizes':<30} {'PINN Error':<15} {'MLP Error':<15} {'SCaSML Error':<15}")
            print("-" * 90)
            for i, gflops in enumerate(gflops_array):
                layer_str = str(layer_sizes_list[i]) if i < len(layer_sizes_list) else "N/A"
                if len(layer_str) > 28:
                    layer_str = layer_str[:25] + "..."
                print(f"{gflops:<15.2e} {layer_str:<30} {pinn_errors[i]:<15.6e} {mlp_errors[i]:<15.6e} {scasml_errors[i]:<15.6e}")
            print()
            
            print("Average Improvement:")
            print(f"  SCaSML vs PINN: {avg_improvement_pinn:+.2f}%")
            print(f"  SCaSML vs MLP: {avg_improvement_mlp:+.2f}%")
            print()
            
            if len(gflops_array) > 0:
                print(f"Highest GFlops Level ({gflops_array[-1]:.2e}):")
                print(f"  PINN error: {pinn_errors[-1]:.6e}")
                print(f"  MLP error: {mlp_errors[-1]:.6e}")
                print(f"  ScaSML error: {scasml_errors[-1]:.6e}")
                print(f"  Improvement vs PINN: {improvements_vs_pinn[-1]:+.2f}%")
                print(f"  Improvement vs MLP: {improvements_vs_mlp[-1]:+.2f}%")
            print("=" * 100)
            
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
        return gflops_levels[-1] if gflops_levels else 1e-6
