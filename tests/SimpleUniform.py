import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import time
import sys
import os
import cProfile
import shutil
import jax.numpy as jnp
from optimizers.Adam import Adam
from matplotlib.colors import LogNorm

# L_inf has been removed

class SimpleUniform(object):
    '''
    Simple Uniform test in high dimensions.

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
        Initializes the simple uniform test with given solvers and equation.

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

    def test(self, save_path, rhomax=2, num_domain=1000, num_boundary=200):
        '''
        Compares solvers on test data after training on a large training dataset.
    
        Parameters:
        save_path (str): The path to save the results.
        rhomax (int): The number of quadrature points for the approximation, equal to the total level
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
        n = rhomax
        # Train solver
        if is_train:
            opt = Adam(eq.n_input,1, self.solver1, eq)
            trained_model1= opt.train(f"{save_path}/model_weights_Adam")
            self.solver1 = trained_model1
            self.solver3.PINN = trained_model1            
        # Generate test data
        data_domain_test, data_boundary_test = eq.generate_test_data(num_domain, num_boundary)
        data_test = jnp.concatenate((data_domain_test, data_boundary_test), axis=0)
        xt_test = data_test[:, :self.dim + 1]
        exact_sol = eq.exact_solution(xt_test)
        errors1 = jnp.zeros(num_domain)
        errors2 = jnp.zeros(num_domain)
        errors3 = jnp.zeros(num_domain)
        rel_error1 = 0
        rel_error2 = 0
        rel_error3 = 0
        real_sol_L2 = 0
        time1, time2, time3 = 0, 0, 0
    
        # Measure the time and predict using solver1
        print("Predicting with solver1 on test data...")
        start = time.time()
        sol1 = self.solver1.predict(data_test)
        time1 += time.time() - start
    
        # Measure the time and predict using solver2
        print("Predicting with solver2 on test data...")
        start = time.time()
        sol2 = self.solver2.u_solve(n, rhomax, data_test)
        time2 += time.time() - start
    
        # Measure the time and predict using solver3
        print("Predicting with solver3 on test data...")
        start = time.time()
        sol3 = self.solver3.u_solve(n, rhomax, data_test)
        time3 += time.time() - start

        # Compute the average error and relative error
        errors1 = jnp.abs(sol1 - exact_sol).flatten()
        errors2 = jnp.abs(sol2 - exact_sol).flatten()
        errors3 = jnp.abs(sol3 - exact_sol).flatten()
        rel_error1 = jnp.linalg.norm(errors1) / jnp.linalg.norm(exact_sol+1e-6)
        rel_error2 = jnp.linalg.norm(errors2) / jnp.linalg.norm(exact_sol+1e-6)
        rel_error3 = jnp.linalg.norm(errors3) / jnp.linalg.norm(exact_sol+1e-6)
        real_sol_L2 = jnp.linalg.norm(exact_sol) / jnp.sqrt(exact_sol.shape[0])
        #stop the profiler
        profiler.disable()
        #save the profiler results
        profiler.dump_stats(f"{save_path}/{eq_name}_rho_{rhomax}.prof")
        #upload the profiler results to wandb
        artifact=wandb.Artifact(f"{eq_name}_rho_{rhomax}", type="profile")
        artifact.add_file(f"{save_path}/{eq_name}_rho_{rhomax}.prof")
        wandb.log_artifact(artifact)
        # open a file to save the output
        log_file = open(f"{save_path}/SimpleUniform.log", "w")
        #redirect stdout and stderr to the log file
        sys.stdout=log_file
        sys.stderr=log_file
        # Print the total time for each solver
        print(f"Total time for PINN: {time1} seconds")
        print(f"Total time for MLP: {time2} seconds")
        print(f"Total time for ScaSML: {time3} seconds")
        wandb.log({"Total time for PINN": time1, "Total time for MLP": time2, "Total time for ScaSML": time3})
        # compute |errors1|-|errors3|,|errrors2|-|errors3|
        errors_13 = errors1 - errors3
        errors_23 = errors2 - errors3
        
        # Calculate comparison metrics
        diff_gp = errors_13
        diff_mlp = errors_23
        
        # Get spatial coordinates (first two dimensions)
        spatial_coords = xt_test[:, :2]

        # =============================================
        # Visualization Configuration
        # =============================================
        COLOR_SCHEME = {
            'PINN': '#000000',     # Black
            'MLP': '#A6A3A4',    # Gray
            'SCaSML': '#2C939A'  # Teal
        }

        # Use DejaVu Sans as a replacement for Arial
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',  # Similar to Arial
            'font.size': 8,
            'axes.labelsize': 9,
            'axes.titlesize': 0,
            'legend.fontsize': 7,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'axes.linewidth': 0.6,
            'lines.linewidth': 0.8,
            'savefig.dpi': 600,
            'savefig.transparent': True,
            'figure.autolayout': False
        })

        # =============================================
        # Figure 1: Error Distribution (Violin Plot)
        # =============================================
        fig1 = plt.figure(figsize=(3.5, 3))
        ax1 = fig1.add_subplot(111)
        
        # Create violin plot
        vp = ax1.violinplot([errors1, errors2, errors3], 
                        showmeans=False, showmedians=True)
        
        # Style violins
        for pc, color in zip(vp['bodies'], COLOR_SCHEME.values()):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.8)
        
        # Configure axes
        ax1.set_yscale('log')
        ax1.set_ylabel('Absolute Error', labelpad=2)
        ax1.set_xticks([1, 2, 3])
        ax1.set_xticklabels(['PINN', 'MLP', 'SCaSML'], rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Final touches
        ax1.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{save_path}/Error_Distribution.pdf", 
                bbox_inches='tight', pad_inches=0.05)
        plt.close()

        # =============================================
        # Figure 2: PINN vs SCaSML Comparison
        # =============================================
        fig2, ax2 = plt.subplots(figsize=(3.5, 3))
        
        # Determine symmetric colorbar limits
        max_diff = max(np.abs(diff_gp).max(), np.abs(diff_mlp).max())
        vmin, vmax = -max_diff, max_diff  # Ensure 0 is centered
        
        # Create hexbin plot
        hb_gp = ax2.hexbin(spatial_coords[:,0], spatial_coords[:,1], 
                        C=diff_gp, cmap='coolwarm', gridsize=30,
                        reduce_C_function=np.mean, mincnt=1,
                        vmin=vmin, vmax=vmax)  # Set symmetric limits
        
        # Add colorbar
        cb_gp = fig2.colorbar(hb_gp, ax=ax2, pad=0.02)
        cb_gp.set_label('Error Difference (PINN - SCaSML)', rotation=270, labelpad=10)
        cb_gp.set_ticks([vmin, 0, vmax])  # Ensure 0 is centered
        
        # Add statistical annotation
        stats_text = (f"Positive count: {np.sum(diff_gp > 0)}\n"
                    f"Negative count: {np.sum(diff_gp < 0)}\n"
                    f"Positive sum: {np.sum(diff_gp[diff_gp > 0]):.2f}\n"
                    f"Negative sum: {np.sum(diff_gp[diff_gp < 0]):.2f}")
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                ha='right', va='top', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Format axes
        ax2.set_xlabel('Spatial Dimension 1', labelpad=2)
        ax2.set_ylabel('Spatial Dimension 2', labelpad=2)
        ax2.set_aspect('equal')
        ax2.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{save_path}/GP_vs_SCaSML.pdf", 
                bbox_inches='tight', pad_inches=0.05)
        plt.close()

        # =============================================
        # Figure 3: MLP vs SCaSML Comparison 
        # =============================================
        fig3, ax3 = plt.subplots(figsize=(3.5, 3))
        
        # Create hexbin plot with same colorbar limits
        hb_mlp = ax3.hexbin(spatial_coords[:,0], spatial_coords[:,1],
                        C=diff_mlp, cmap='coolwarm', gridsize=30,
                        reduce_C_function=np.mean, mincnt=1,
                        vmin=vmin, vmax=vmax)  # Use same limits as PINN plot
        
        # Add colorbar
        cb_mlp = fig3.colorbar(hb_mlp, ax=ax3, pad=0.02)
        cb_mlp.set_label('Error Difference (MLP - SCaSML)', rotation=270, labelpad=10)
        cb_mlp.set_ticks([vmin, 0, vmax])  # Ensure 0 is centered
        
        # Add statistical annotation
        stats_text = (f"Positive count: {np.sum(diff_mlp > 0)}\n"
                    f"Negative count: {np.sum(diff_mlp < 0)}\n"
                    f"Positive sum: {np.sum(diff_mlp[diff_mlp > 0]):.2f}\n"
                    f"Negative sum: {np.sum(diff_mlp[diff_mlp < 0]):.2f}")
        ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
                ha='right', va='top', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Format axes
        ax3.set_xlabel('Spatial Dimension 1', labelpad=2)
        ax3.set_ylabel('Spatial Dimension 2', labelpad=2)
        ax3.set_aspect('equal')
        ax3.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{save_path}/MLP_vs_SCaSML.pdf", 
                bbox_inches='tight', pad_inches=0.05)
        plt.close()
        
        # =============================================
        # Figure 4: Relative L2 Improvement Bar Plot
        # =============================================
        # Calculate mean relative errors
        mean_rel_error1 = np.mean(rel_error1)
        mean_rel_error2 = np.mean(rel_error2)
        mean_rel_error3 = np.mean(rel_error3)

        # Calculate improvements
        improvement_gp = (mean_rel_error1 - mean_rel_error3) / mean_rel_error1 * 100
        improvement_mlp = (mean_rel_error2 - mean_rel_error3) / mean_rel_error2 * 100

        # Create bar plot
        fig, ax = plt.subplots(figsize=(3.5, 3))
        methods = ['PINN', 'MLP', 'SCaSML']
        colors = ['#000000', '#A6A3A4', '#2C939A']
        
        # Plot bars
        bars = ax.bar(methods, [mean_rel_error1, mean_rel_error2, mean_rel_error3], 
                    color=colors, edgecolor='black')
        
        # Annotate improvements
        def format_improvement(val):
            if val > 0: return f"-{val:.1f}%"
            elif val < 0: return f"+{abs(val):.1f}%"
            else: return "0%"
        
        ax.text(0, mean_rel_error1*1.05, format_improvement(improvement_gp),
            ha='center', va='bottom', fontsize=7)
        ax.text(1, mean_rel_error2*1.05, format_improvement(improvement_mlp),
            ha='center', va='bottom', fontsize=7)

        # Formatting
        ax.set_ylabel('Mean Relative L2 Error')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{save_path}/Relative_L2_Improvement.pdf", bbox_inches='tight')
        plt.close()
    
        # =============================================
        # Figure 5: Spatiotemporal Error Analysis
        # =============================================
        # Compute (x1, x2) from test data
        x1_values = xt_test[:, 0]  # First dimension is x1
        x2_values = xt_test[:, 1]  # Second dimension is x2

        # Create grid parameters
        x1_grid_num, x2_grid_num = 3, 3  # mesh density
        x1_bins = np.linspace(np.min(x1_values), np.max(x1_values), x1_grid_num + 1)
        x2_bins = np.linspace(np.min(x2_values), np.max(x2_values), x2_grid_num + 1)

        # Initialize error arrays for grid cells
        errors1_grid = np.zeros((x2_grid_num, x1_grid_num))
        errors2_grid = np.zeros((x2_grid_num, x1_grid_num))
        errors3_grid = np.zeros((x2_grid_num, x1_grid_num))

        # Compute errors for each grid cell
        for i in range(x2_grid_num):
            for j in range(x1_grid_num):
                # Find points in current cell
                x1_mask = (x1_values >= x1_bins[j]) & (x1_values < x1_bins[j+1])
                x2_mask = (x2_values >= x2_bins[i]) & (x2_values < x2_bins[i+1])
                cell_mask = x1_mask & x2_mask
                
                if np.sum(cell_mask) > 0:
                    # Compute mean L1 errors
                    errors1_grid[i, j] = np.mean(errors1[cell_mask])
                    errors2_grid[i, j] = np.mean(errors2[cell_mask])
                    errors3_grid[i, j] = np.mean(errors3[cell_mask])

        # =============================================
        # New Code: Spatiotemporal Error Visualization
        # =============================================
        def plot_spatio_temp_error(data, label, filename):
            fig, ax = plt.subplots(figsize=(5, 4))
            norm = LogNorm(vmin=1e-6, vmax=np.max([errors1.max(), errors2.max(), errors3.max()]))
            
            # Create heatmap
            im = ax.pcolormesh(x1_bins, x2_bins, data, cmap='viridis', norm=norm, shading='auto')
            
            # Add annotations
            for i in range(x2_grid_num):
                for j in range(x1_grid_num):
                    if data[i, j] > 0:
                        x1_center = (x1_bins[j] + x1_bins[j+1])/2
                        x2_center = (x2_bins[i] + x2_bins[i+1])/2
                        ax.text(x1_center, x2_center, f'{data[i, j]:.2e}', 
                                ha='center', va='center', color='black', fontsize=6)
            
            # Formatting
            cb = fig.colorbar(im, ax=ax, label='L1 Error (log scale)')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            plt.tight_layout()
            plt.savefig(f"{save_path}/{filename}.pdf", bbox_inches='tight')
            plt.close()

        # Generate three spatiotemporal plots
        plot_spatio_temp_error(errors1_grid, 'PINN', 'GP_Spatiotemporal_Errors')
        plot_spatio_temp_error(errors2_grid, 'MLP', 'MLP_Spatiotemporal_Errors')
        plot_spatio_temp_error(errors3_grid, 'SCaSML', 'SCaSML_Spatiotemporal_Errors')

        # Print the results
        print(f"PINN rel L2, rho={rhomax}->", rel_error1)
        
        
        print(f"MLP rel L2, rho={rhomax}->", rel_error2)
        
        
        print(f"ScaSML rel L2, rho={rhomax}->", rel_error3)
        
        
        print("Real Solution->", real_sol_L2)
        

        print(f"PINN L1, rho={rhomax}->","min:", jnp.min(errors1), "max:", jnp.max(errors1), "mean:", jnp.mean(errors1))
        
        
        print(f"MLP L1, rho={rhomax}->","min:", jnp.min(errors2), "max:", jnp.max(errors2), "mean:", jnp.mean(errors2))
        
        
        print(f"ScaSML L1, rho={rhomax}->","min:", jnp.min(errors3), "max:", jnp.max(errors3), "mean:", jnp.mean(errors3))
        
        
        # Calculate the sums of positive and negative differences
        positive_sum_13 = jnp.sum(errors_13[errors_13 > 0])
        negative_sum_13 = jnp.sum(errors_13[errors_13 < 0])
        positive_sum_23 = jnp.sum(errors_23[errors_23 > 0])
        negative_sum_23 = jnp.sum(errors_23[errors_23 < 0])
        # Display the positive count, negative count, positive sum, and negative sum of the difference of the errors
        print(f'PINN L2 - ScaSML L2, rho={rhomax}->','positive count:', jnp.sum(errors_13 > 0), 'negative count:', jnp.sum(errors_13 < 0), 'positive sum:', positive_sum_13, 'negative sum:', negative_sum_13)
        print(f'MLP L2 - ScaSML L2, rho={rhomax}->','positive count:', jnp.sum(errors_23 > 0), 'negative count:', jnp.sum(errors_23 < 0), 'positive sum:', positive_sum_23, 'negative sum:', negative_sum_23)
        # Log the results to wandb
        wandb.log({f"mean of PINN L2, rho={rhomax}": jnp.mean(errors1), f"mean of MLP L2, rho={rhomax}": jnp.mean(errors2), f"mean of ScaSML L2, rho={rhomax}": jnp.mean(errors3)})
        wandb.log({f"min of PINN L2, rho={rhomax}": jnp.min(errors1), f"min of MLP L2, rho={rhomax}": jnp.min(errors2), f"min of ScaSML L2, rho={rhomax}": jnp.min(errors3)})
        wandb.log({f"max of PINN L2, rho={rhomax}": jnp.max(errors1), f"max of MLP L2, rho={rhomax}": jnp.max(errors2), f"max of ScaSML L2, rho={rhomax}": jnp.max(errors3)})
        wandb.log({f"positive count of PINN L2 - ScaSML L2, rho={rhomax}": jnp.sum(errors_13 > 0), f"negative count of PINN L2 - ScaSML L2, rho={rhomax}": jnp.sum(errors_13 < 0), f"positive sum of PINN L2 - ScaSML L2, rho={rhomax}": positive_sum_13, f"negative sum of PINN L2 - ScaSML L2, rho={rhomax}": negative_sum_13})
        wandb.log({f"positive count of MLP L2 - ScaSML L2, rho={rhomax}": jnp.sum(errors_23 > 0), f"negative count of MLP L2 - ScaSML L2, rho={rhomax}": jnp.sum(errors_23 < 0), f"positive sum of MLP L2 - ScaSML L2, rho={rhomax}": positive_sum_23, f"negative sum of MLP L2 - ScaSML L2, rho={rhomax}": negative_sum_23})
        # reset stdout and stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        #close the log file
        log_file.close()
        return rhomax
