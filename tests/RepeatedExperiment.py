import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import time
import sys
import os
import cProfile
from optimizers.Adam import Adam
from scipy import stats

class RepeatedExperiment(object):
    '''
    Repeated experiment test that computes statistics over multiple runs.
    
    Instead of computing statistics over individual test points, this class
    runs the experiment multiple times and computes mean ± std and confidence
    intervals for the mean relative L2 error, mean L2 error, and mean L1 error
    across repetitions.

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
        Initializes the repeated experiment test with given solvers and equation.

        Parameters:
        equation (object): The equation object containing problem specifics.
        solver1 (object): The PINN solver.
        solver2 (object): The MLP solver object.
        solver3 (object): The ScaSML solver object.
        is_train (bool): Whether to train the PINN solver.
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

    def run_single_experiment(self, rhomax, num_domain, num_boundary, seed=None):
        '''
        Run a single experiment with given parameters.
        
        Parameters:
        rhomax (int): The number of quadrature points for the approximation.
        num_domain (int): The number of points in the test domain.
        num_boundary (int): The number of points on the test boundary.
        seed (int): Random seed for reproducibility.
        
        Returns:
        dict: Dictionary containing error metrics for this run.
        '''
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        eq = self.equation
        n = rhomax
        
        # Generate test data
        data_domain_test, data_boundary_test = eq.generate_test_data(num_domain, num_boundary)
        xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
        exact_sol = eq.exact_solution(xt_test)
        
        # Measure the time and predict using solver1
        start = time.time()
        sol1 = self.solver1.predict(xt_test)
        time1 = time.time() - start
        
        # Measure the time and predict using solver2
        start = time.time()
        sol2 = self.solver2.u_solve(n, rhomax, xt_test)
        time2 = time.time() - start
        
        # Measure the time and predict using solver3
        start = time.time()
        sol3 = self.solver3.u_solve(n, rhomax, xt_test)
        time3 = time.time() - start
        
        # Create mask for valid data points
        valid_mask = ~(np.isnan(sol1) | np.isnan(sol2) | np.isnan(sol3) | np.isnan(exact_sol)).flatten()
        
        if np.sum(valid_mask) == 0:
            print("Warning: All predictions are NaN in this run.")
            return None
        
        # Calculate absolute and squared errors
        sol1_valid = sol1.flatten()[valid_mask]
        sol2_valid = sol2.flatten()[valid_mask]
        sol3_valid = sol3.flatten()[valid_mask]
        exact_sol_valid = exact_sol.flatten()[valid_mask]
        
        diff1 = sol1_valid - exact_sol_valid
        diff2 = sol2_valid - exact_sol_valid
        diff3 = sol3_valid - exact_sol_valid
        
        errors1 = np.abs(diff1)
        errors2 = np.abs(diff2)
        errors3 = np.abs(diff3)
        
        errors1_l2 = diff1 ** 2
        errors2_l2 = diff2 ** 2
        errors3_l2 = diff3 ** 2
        
        # Calculate mean errors (across all test points in this run)
        mean_l1_error1 = np.mean(errors1)
        mean_l1_error2 = np.mean(errors2)
        mean_l1_error3 = np.mean(errors3)
        
        mean_l2_error1 = np.mean(errors1_l2)
        mean_l2_error2 = np.mean(errors2_l2)
        mean_l2_error3 = np.mean(errors3_l2)
        
        # Calculate relative L2 errors
        rel_error1 = np.linalg.norm(errors1) / np.linalg.norm(exact_sol_valid)
        rel_error2 = np.linalg.norm(errors2) / np.linalg.norm(exact_sol_valid)
        rel_error3 = np.linalg.norm(errors3) / np.linalg.norm(exact_sol_valid)
        
        return {
            'mean_l1_error1': mean_l1_error1,
            'mean_l1_error2': mean_l1_error2,
            'mean_l1_error3': mean_l1_error3,
            'mean_l2_error1': mean_l2_error1,
            'mean_l2_error2': mean_l2_error2,
            'mean_l2_error3': mean_l2_error3,
            'rel_error1': rel_error1,
            'rel_error2': rel_error2,
            'rel_error3': rel_error3,
            'time1': time1,
            'time2': time2,
            'time3': time3
        }

    def test(self, save_path, rhomax=2, num_domain=1000, num_boundary=200, 
             num_repetitions=10, start_seed=42):
        '''
        Run repeated experiments and compute statistics across repetitions.
        
        Parameters:
        save_path (str): The path to save the results.
        rhomax (int): The number of quadrature points for the approximation.
        num_domain (int): The number of points in the test domain.
        num_boundary (int): The number of points on the test boundary.
        num_repetitions (int): Number of times to repeat the experiment.
        start_seed (int): Starting random seed for reproducibility.
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
        
        # Train solver once (if needed)
        if is_train:
            print("Training PINN solver...")
            opt = Adam(eq.n_input, 1, self.solver1, eq)
            trained_model1 = opt.train(f"{save_path}/model_weights_Adam")
            self.solver1 = trained_model1
            self.solver3.PINN = trained_model1
        
        # Storage for results across repetitions
        results = {
            'mean_l1_error1': [],
            'mean_l1_error2': [],
            'mean_l1_error3': [],
            'mean_l2_error1': [],
            'mean_l2_error2': [],
            'mean_l2_error3': [],
            'rel_error1': [],
            'rel_error2': [],
            'rel_error3': [],
            'time1': [],
            'time2': [],
            'time3': []
        }
        
        # Run repeated experiments
        print(f"\nRunning {num_repetitions} repeated experiments...")
        for i in range(num_repetitions):
            print(f"Experiment {i+1}/{num_repetitions}...")
            seed = start_seed + i
            result = self.run_single_experiment(rhomax, num_domain, num_boundary, seed)
            
            if result is not None:
                for key in results.keys():
                    results[key].append(result[key])
        
        # Convert to numpy arrays
        for key in results.keys():
            results[key] = np.array(results[key])
        
        # Stop the profiler
        profiler.disable()
        profiler.dump_stats(f"{save_path}/{eq_name}_rho_{rhomax}_repeated.prof")
        
        # Upload the profiler results to wandb
        artifact = wandb.Artifact(f"{eq_name}_rho_{rhomax}_repeated", type="profile")
        artifact.add_file(f"{save_path}/{eq_name}_rho_{rhomax}_repeated.prof")
        wandb.log_artifact(artifact)
        
        # Open a file to save the output
        log_file = open(f"{save_path}/RepeatedExperiment.log", "w")
        # Redirect stdout and stderr to the log file
        sys.stdout = log_file
        sys.stderr = log_file
        
        # Calculate statistics across repetitions
        print(f"Results over {len(results['rel_error1'])} successful repetitions:")
        print("=" * 80)
        
        # Helper function to compute stats
        def compute_stats(data, metric_name, solver_name):
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=1)  # Sample std
            ci_95 = 1.96 * std_val / np.sqrt(len(data))
            min_val = np.min(data)
            max_val = np.max(data)
            
            print(f"\n{solver_name} - {metric_name}:")
            print(f"  Mean: {mean_val:.6e}")
            print(f"  Std:  {std_val:.6e}")
            print(f"  95% CI: ±{ci_95:.6e}")
            print(f"  Range: [{min_val:.6e}, {max_val:.6e}]")
            
            return mean_val, std_val, ci_95, min_val, max_val
        
        # Compute statistics for each metric and solver
        print("\n" + "=" * 80)
        print("MEAN RELATIVE L2 ERROR (across repetitions)")
        print("=" * 80)
        
        pinn_rel_l2_stats = compute_stats(results['rel_error1'], "Mean Relative L2 Error", "PINN")
        mlp_rel_l2_stats = compute_stats(results['rel_error2'], "Mean Relative L2 Error", "MLP")
        scasml_rel_l2_stats = compute_stats(results['rel_error3'], "Mean Relative L2 Error", "SCaSML")
        
        print("\n" + "=" * 80)
        print("MEAN L1 ERROR (across repetitions)")
        print("=" * 80)
        
        pinn_l1_stats = compute_stats(results['mean_l1_error1'], "Mean L1 Error", "PINN")
        mlp_l1_stats = compute_stats(results['mean_l1_error2'], "Mean L1 Error", "MLP")
        scasml_l1_stats = compute_stats(results['mean_l1_error3'], "Mean L1 Error", "SCaSML")
        
        print("\n" + "=" * 80)
        print("MEAN L2 ERROR (squared, across repetitions)")
        print("=" * 80)
        
        pinn_l2_stats = compute_stats(results['mean_l2_error1'], "Mean L2 Error", "PINN")
        mlp_l2_stats = compute_stats(results['mean_l2_error2'], "Mean L2 Error", "MLP")
        scasml_l2_stats = compute_stats(results['mean_l2_error3'], "Mean L2 Error", "SCaSML")
        
        # Perform paired t-tests
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE TESTS (Paired t-test)")
        print("=" * 80)
        
        print("\nMean Relative L2 Error:")
        t_stat, p_val = stats.ttest_rel(results['rel_error1'], results['rel_error3'])
        print(f"  PINN vs SCaSML: t={t_stat:.6f}, p={p_val:.6e}")
        
        t_stat, p_val = stats.ttest_rel(results['rel_error2'], results['rel_error3'])
        print(f"  MLP vs SCaSML: t={t_stat:.6f}, p={p_val:.6e}")
        
        t_stat, p_val = stats.ttest_rel(results['rel_error1'], results['rel_error2'])
        print(f"  PINN vs MLP: t={t_stat:.6f}, p={p_val:.6e}")
        
        print("\nMean L1 Error:")
        t_stat, p_val = stats.ttest_rel(results['mean_l1_error1'], results['mean_l1_error3'])
        print(f"  PINN vs SCaSML: t={t_stat:.6f}, p={p_val:.6e}")
        
        t_stat, p_val = stats.ttest_rel(results['mean_l1_error2'], results['mean_l1_error3'])
        print(f"  MLP vs SCaSML: t={t_stat:.6f}, p={p_val:.6e}")
        
        t_stat, p_val = stats.ttest_rel(results['mean_l1_error1'], results['mean_l1_error2'])
        print(f"  PINN vs MLP: t={t_stat:.6f}, p={p_val:.6e}")
        
        print("\nMean L2 Error:")
        t_stat, p_val = stats.ttest_rel(results['mean_l2_error1'], results['mean_l2_error3'])
        print(f"  PINN vs SCaSML: t={t_stat:.6f}, p={p_val:.6e}")
        
        t_stat, p_val = stats.ttest_rel(results['mean_l2_error2'], results['mean_l2_error3'])
        print(f"  MLP vs SCaSML: t={t_stat:.6f}, p={p_val:.6e}")
        
        t_stat, p_val = stats.ttest_rel(results['mean_l2_error1'], results['mean_l2_error2'])
        print(f"  PINN vs MLP: t={t_stat:.6f}, p={p_val:.6e}")
        
        # Timing statistics
        print("\n" + "=" * 80)
        print("TIMING STATISTICS")
        print("=" * 80)
        
        compute_stats(results['time1'], "Execution Time (seconds)", "PINN")
        compute_stats(results['time2'], "Execution Time (seconds)", "MLP")
        compute_stats(results['time3'], "Execution Time (seconds)", "SCaSML")
        
        # Log to wandb
        print("\n" + "=" * 80)
        print("Logging results to Weights & Biases...")
        print("=" * 80)
        
        # Log mean relative L2 error statistics
        wandb.log({
            f"rep_mean_rel_l2_pinn_mean, rho={rhomax}": pinn_rel_l2_stats[0],
            f"rep_mean_rel_l2_pinn_std, rho={rhomax}": pinn_rel_l2_stats[1],
            f"rep_mean_rel_l2_pinn_ci, rho={rhomax}": pinn_rel_l2_stats[2],
            f"rep_mean_rel_l2_mlp_mean, rho={rhomax}": mlp_rel_l2_stats[0],
            f"rep_mean_rel_l2_mlp_std, rho={rhomax}": mlp_rel_l2_stats[1],
            f"rep_mean_rel_l2_mlp_ci, rho={rhomax}": mlp_rel_l2_stats[2],
            f"rep_mean_rel_l2_scasml_mean, rho={rhomax}": scasml_rel_l2_stats[0],
            f"rep_mean_rel_l2_scasml_std, rho={rhomax}": scasml_rel_l2_stats[1],
            f"rep_mean_rel_l2_scasml_ci, rho={rhomax}": scasml_rel_l2_stats[2]
        })
        
        # Log mean L1 error statistics
        wandb.log({
            f"rep_mean_l1_pinn_mean, rho={rhomax}": pinn_l1_stats[0],
            f"rep_mean_l1_pinn_std, rho={rhomax}": pinn_l1_stats[1],
            f"rep_mean_l1_pinn_ci, rho={rhomax}": pinn_l1_stats[2],
            f"rep_mean_l1_mlp_mean, rho={rhomax}": mlp_l1_stats[0],
            f"rep_mean_l1_mlp_std, rho={rhomax}": mlp_l1_stats[1],
            f"rep_mean_l1_mlp_ci, rho={rhomax}": mlp_l1_stats[2],
            f"rep_mean_l1_scasml_mean, rho={rhomax}": scasml_l1_stats[0],
            f"rep_mean_l1_scasml_std, rho={rhomax}": scasml_l1_stats[1],
            f"rep_mean_l1_scasml_ci, rho={rhomax}": scasml_l1_stats[2]
        })
        
        # Log mean L2 error statistics
        wandb.log({
            f"rep_mean_l2_pinn_mean, rho={rhomax}": pinn_l2_stats[0],
            f"rep_mean_l2_pinn_std, rho={rhomax}": pinn_l2_stats[1],
            f"rep_mean_l2_pinn_ci, rho={rhomax}": pinn_l2_stats[2],
            f"rep_mean_l2_mlp_mean, rho={rhomax}": mlp_l2_stats[0],
            f"rep_mean_l2_mlp_std, rho={rhomax}": mlp_l2_stats[1],
            f"rep_mean_l2_mlp_ci, rho={rhomax}": mlp_l2_stats[2],
            f"rep_mean_l2_scasml_mean, rho={rhomax}": scasml_l2_stats[0],
            f"rep_mean_l2_scasml_std, rho={rhomax}": scasml_l2_stats[1],
            f"rep_mean_l2_scasml_ci, rho={rhomax}": scasml_l2_stats[2]
        })
        
        # Create visualization
        self._create_visualizations(results, save_path, rhomax)
        
        # Reset stdout and stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        log_file.close()
        
        print(f"\nResults saved to {save_path}")
        return rhomax

    def _create_visualizations(self, results, save_path, rhomax):
        '''
        Create visualization plots for the repeated experiment results.
        
        Parameters:
        results (dict): Dictionary containing results from all repetitions.
        save_path (str): Path to save the plots.
        rhomax (int): The rhomax value used in experiments.
        '''
        COLOR_SCHEME = {
            'PINN': '#000000',     # Black
            'MLP': '#A6A3A4',      # Gray
            'SCaSML': '#2C939A'    # Teal
        }
        
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 8,
            'axes.labelsize': 9,
            'axes.titlesize': 10,
            'legend.fontsize': 7,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'axes.linewidth': 0.6,
            'lines.linewidth': 0.8,
            'savefig.dpi': 600,
            'savefig.transparent': True
        })
        
        # Figure 1: Mean Relative L2 Error across repetitions
        fig, ax = plt.subplots(figsize=(4, 3))
        
        methods = ['PINN', 'MLP', 'SCaSML']
        means = [np.mean(results['rel_error1']), 
                 np.mean(results['rel_error2']), 
                 np.mean(results['rel_error3'])]
        stds = [np.std(results['rel_error1'], ddof=1),
                np.std(results['rel_error2'], ddof=1),
                np.std(results['rel_error3'], ddof=1)]
        
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=[COLOR_SCHEME[m] for m in methods],
                     edgecolor='black', alpha=0.8)
        
        ax.set_ylabel('Mean Relative L2 Error')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/Mean_Relative_L2_Error_Repetitions.pdf", bbox_inches='tight')
        plt.close()
        
        # Figure 2: Mean L1 Error across repetitions
        fig, ax = plt.subplots(figsize=(4, 3))
        
        means = [np.mean(results['mean_l1_error1']), 
                 np.mean(results['mean_l1_error2']), 
                 np.mean(results['mean_l1_error3'])]
        stds = [np.std(results['mean_l1_error1'], ddof=1),
                np.std(results['mean_l1_error2'], ddof=1),
                np.std(results['mean_l1_error3'], ddof=1)]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=[COLOR_SCHEME[m] for m in methods],
                     edgecolor='black', alpha=0.8)
        
        ax.set_ylabel('Mean L1 Error')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/Mean_L1_Error_Repetitions.pdf", bbox_inches='tight')
        plt.close()
        
        # Figure 3: Mean L2 Error across repetitions
        fig, ax = plt.subplots(figsize=(4, 3))
        
        means = [np.mean(results['mean_l2_error1']), 
                 np.mean(results['mean_l2_error2']), 
                 np.mean(results['mean_l2_error3'])]
        stds = [np.std(results['mean_l2_error1'], ddof=1),
                np.std(results['mean_l2_error2'], ddof=1),
                np.std(results['mean_l2_error3'], ddof=1)]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=[COLOR_SCHEME[m] for m in methods],
                     edgecolor='black', alpha=0.8)
        
        ax.set_ylabel('Mean L2 Error (Squared)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/Mean_L2_Error_Repetitions.pdf", bbox_inches='tight')
        plt.close()
        
        # Figure 4: Box plots for all three metrics
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        
        # Relative L2 Error
        data_rel = [results['rel_error1'], results['rel_error2'], results['rel_error3']]
        bp1 = axes[0].boxplot(data_rel, labels=methods, patch_artist=True)
        for patch, color in zip(bp1['boxes'], [COLOR_SCHEME[m] for m in methods]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        axes[0].set_ylabel('Mean Relative L2 Error')
        axes[0].grid(axis='y', linestyle='--', alpha=0.4)
        axes[0].spines[['top', 'right']].set_visible(False)
        
        # L1 Error
        data_l1 = [results['mean_l1_error1'], results['mean_l1_error2'], results['mean_l1_error3']]
        bp2 = axes[1].boxplot(data_l1, labels=methods, patch_artist=True)
        for patch, color in zip(bp2['boxes'], [COLOR_SCHEME[m] for m in methods]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        axes[1].set_ylabel('Mean L1 Error')
        axes[1].grid(axis='y', linestyle='--', alpha=0.4)
        axes[1].spines[['top', 'right']].set_visible(False)
        
        # L2 Error
        data_l2 = [results['mean_l2_error1'], results['mean_l2_error2'], results['mean_l2_error3']]
        bp3 = axes[2].boxplot(data_l2, labels=methods, patch_artist=True)
        for patch, color in zip(bp3['boxes'], [COLOR_SCHEME[m] for m in methods]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        axes[2].set_ylabel('Mean L2 Error (Squared)')
        axes[2].grid(axis='y', linestyle='--', alpha=0.4)
        axes[2].spines[['top', 'right']].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/All_Errors_Boxplots.pdf", bbox_inches='tight')
        plt.close()
