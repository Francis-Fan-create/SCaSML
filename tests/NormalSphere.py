import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm
import time
import sys
import os
import cProfile
import shutil
from optimizers.Adam_LBFGS import Adam_LBFGS
from optimizers.L_inf import L_inf
import jax.numpy as jnp

class NormalSphere(object):
    '''
    Normal sphere test in high dimensions.

    Attributes:
    equation (object): An object representing the equation to solve.
    dim (int): The dimension of the input space minus one.
    solver1 (object): A jax Gaussian Process model.
    solver2 (object): An object for the MLP solver.
    solver3 (object): An object for the ScaSML solver.
    t0 (float): The initial time.
    T (float): The final time.
    radius (float): The radius of the sphere calculated based on the dimension and time.
    '''
    def __init__(self, equation, solver1, solver2, solver3, is_train):
        '''
        Initializes the normal spheres with given solvers and equation.

        Parameters:
        equation (object): The equation object containing problem specifics.
        solver1 (object): The PINN solver.
        solver2 (object): The MLP solver object.
        solver3 (object): The ScaSML solver object.
        '''
        #save original stdout and stderr
        self.stdout=sys.stdout
        self.stderr=sys.stderr
        # Initialize the normal spheres
        self.equation = equation
        self.equation.test_geometry()
        self.dim = equation.n_input - 1  # equation.n_input: int
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.t0 = equation.t0  # equation.t0: float
        self.T = equation.T  # equation.T: float
        self.test_T = equation.test_T
        self.radius = jnp.sqrt(self.dim * (self.test_T - self.t0) ** 2)  # radius: float, calculated based on dimension and time
        self.is_train = is_train 
    
    def test(self, save_path, rhomax=2, n_samples=100, x_grid_num=100, t_grid_num=10):
        '''
        Compares solvers on different distances on the sphere.
    
        Parameters:
        save_path (str): The path to save the results.
        rhomax (int): The number of quadrature points for the approximation, equal to the total level of the solver.
        n_samples (int): The number of samples for testing.
        x_grid_num (int): The number of grid points in the x dimension.
        t_grid_num (int): The number of grid points in the time dimension.
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
        eq = self.equation
        eq_name = eq.__class__.__name__
        n = rhomax
        # Generate training data
        x_grid = jnp.linspace(0, self.radius, x_grid_num)
        t_grid = jnp.linspace(self.t0, self.test_T , t_grid_num) # Adjust the time grid for testing
        x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)
        errors1 = jnp.zeros_like(x_mesh)
        errors2 = jnp.zeros_like(x_mesh)
        errors3 = jnp.zeros_like(x_mesh)
        rel_error1 = jnp.zeros_like(x_mesh)
        rel_error2 = jnp.zeros_like(x_mesh)
        rel_error3 = jnp.zeros_like(x_mesh)
        real_sol_abs = jnp.zeros_like(x_mesh)
        time1, time2, time3 = 0, 0, 0


        # Train solver
        if is_train:
            domain_size = 100
            data = eq.generate_data(domain_size)
            opt1 = Adam_LBFGS(eq.n_input,1, self.solver1, data, eq)
            trained_model1=opt1.train(f"{save_path}/model_weights_Adam_LBFGS.params")
            trained_net1= trained_model1.net
            opt2 = L_inf(eq.n_input,1, trained_net1, data, eq)
            trained_model2=opt2.train(f"{save_path}/model_weights_L_inf.params")
            trained_net2= trained_model2.net
            self.solver1 = trained_net2
            self.solver3.PINN = trained_net2

        # Compute the errors
        for i in tqdm(range(x_mesh.shape[0]), desc="Computing errors"):
            for j in tqdm(range(x_mesh.shape[1]), desc=f"Computing errors at time {t_grid[i]}"):
                x_values = jnp.random.normal(0, 1, (n_samples, self.dim))
                x_values /= jnp.linalg.norm(x_values, axis=1)[:, jnp.newaxis]
                x_values *= x_mesh[i, j]
                t_values = jnp.full((n_samples, 1), t_mesh[i, j])
                xt_values = jnp.concatenate((x_values, t_values), axis=1)
                exact_sol = eq.exact_solution(xt_values)

                # Predict with solver1
                start = time.time()
                sol1 = self.solver1(xt_values)
                time1 += time.time() - start

                # Measure the time for solver2
                start = time.time()
                sol2 = self.solver2.u_solve(n, rhomax, xt_values)
                time2 += time.time() - start

                # Predict with solver3
                start = time.time()
                sol3 = self.solver3.u_solve(n, rhomax, xt_values)
                time3 += time.time() - start

                # Compute the average error and relative error
                errors1[i, j] += jnp.linalg.norm(sol1 - exact_sol)
                errors2[i, j] += jnp.linalg.norm(sol2 - exact_sol)
                errors3[i, j] += jnp.linalg.norm(sol3 - exact_sol)
                rel_error1[i, j] += jnp.linalg.norm(sol1 - exact_sol) / (jnp.linalg.norm(exact_sol))
                rel_error2[i, j] += jnp.linalg.norm(sol2 - exact_sol) / (jnp.linalg.norm(exact_sol))
                rel_error3[i, j] += jnp.linalg.norm(sol3 - exact_sol) / (jnp.linalg.norm(exact_sol))
                real_sol_abs[i, j] = jnp.linalg.norm(exact_sol)

        #stop the profiler
        profiler.disable()
        #save the profiler results
        profiler.dump_stats(f"{save_path}/{eq_name}_rho_{rhomax}.prof")
        #upload the profiler results to wandb
        artifact=wandb.Artifact(f"{eq_name}_rho_{rhomax}", type="profile")
        artifact.add_file(f"{save_path}/{eq_name}_rho_{rhomax}.prof")
        wandb.log_artifact(artifact)
        # open a file to save the output
        log_file = open(f"{save_path}/NormalSphere.log", "w")
        #redirect stdout and stderr to the log file
        sys.stdout=log_file
        sys.stderr=log_file
        # Print the total time for each solver
        print(f"Total time for PINN: {time1} seconds")
        print(f"Total time for MLP: {time2} seconds")
        print(f"Total time for ScaSML: {time3} seconds")
        wandb.log({"Total time for PINN": time1, "Total time for MLP": time2, "Total time for ScaSML": time3})
        # compute |errors1|-|errors3|,|errrors2|-|errors3|,|errors1|-|errors2|
        errors_13=errors1-errors3
        errors_23=errors2-errors3
        errors_12=errors1-errors2
        
        plt.figure()
        # collect all absolute errors
        errors = [errors1.flatten(), errors2.flatten(), errors3.flatten()]
        # Create a boxplot
        plt.boxplot(errors, labels=['PINN_L2', 'MLP_L2', 'ScaSML_L2'])
        plt.xticks(rotation=45)
        plt.yscale('log')
        # Add a title and labels
        plt.title('L2 Error Distribution')
        plt.ylabel('L2 Error Value')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f"{save_path}/L2_Error_Distribution.png")
        # Upload the plot to wandb
        wandb.log({"Error Distribution": wandb.Image(f"{save_path}/L2_Error_Distribution.png")})

        plt.figure()
        # collect all absolute errors
        errors = [errors1.flatten(), errors2.flatten(), errors3.flatten()]
        # Calculate means and standard deviations
        means = [jnp.mean(e) for e in errors]
        stds = [jnp.std(e) for e in errors]
        # Define labels
        labels = ['PINN_L2', 'MLP_L2', 'ScaSML_L2']
        x_pos = range(len(labels))
        # Create an error bar plot
        plt.errorbar(x_pos, means, yerr=stds, capsize=5, capthick=2, ecolor='black',  marker='s', markersize=7, mfc='red', mec='black')
        plt.xticks(x_pos, labels, rotation=45)
        plt.yscale('log')
        # Add a title and labels
        plt.title('L2 Error Distribution')
        plt.ylabel('L2 Error Value')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f"{save_path}/L2_Error_Distribution_errorbar.png")
        # Upload the plot to wandb
        wandb.log({"Error Distribution": wandb.Image(f"{save_path}/L2_Error_Distribution_errorbar.png")})

        plt.figure()
        #collect all relative errors
        rel_errors = [rel_error1.flatten(), rel_error2.flatten(), rel_error3.flatten()]
        # Create a boxplot
        plt.boxplot(rel_errors, labels=['PINN_L2', 'MLP_L2', 'ScaSML_L2'])
        plt.xticks(rotation=45)
        plt.yscale('log')
        # Add a title and labels
        plt.title('Relative Error Distribution')
        plt.ylabel('Relative Error Value')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f"{save_path}/Relative_Error_Distribution.png")
        # Upload the plot to wandb
        wandb.log({"Relative Error Distribution": wandb.Image(f"{save_path}/Relative_Error_Distribution.png")})

        plt.figure()
        # Collect all relative errors
        rel_errors = [rel_error1.flatten(), rel_error2.flatten(), rel_error3.flatten()]
        # Calculate means and standard deviations for each group
        means = [jnp.mean(errors) for errors in rel_errors]
        stds = [jnp.std(errors) for errors in rel_errors]
        # Define labels for each group
        labels = ['PINN_L2', 'MLP_L2', 'ScaSML_L2']
        x_pos = range(len(labels))
        # Create an error bar plot
        plt.errorbar(x_pos, means, yerr=stds, capsize=5, capthick=2, ecolor='black',  marker='s', markersize=7, mfc='red', mec='black')
        # Set the x-ticks to use the labels and rotate them for better readability
        plt.xticks(x_pos, labels, rotation=45)
        # Set the y-axis to use a logarithmic scale
        plt.yscale('log')
        # Add a title and labels to the plot
        plt.title('Relative Error Distribution')
        plt.ylabel('Relative Error Value')
        # Adjust layout for better display
        plt.tight_layout()
        # Save the plot to a file
        plt.savefig(f"{save_path}/Relative_Error_Distribution_errorbar.png")
        # Upload the plot to wandb
        wandb.log({"Relative Error Distribution": wandb.Image(f"{save_path}/Relative_Error_Distribution_errorbar.png")})    

        #find the global minimum and maximum relative error
        vmin = min(jnp.min(rel_error1), jnp.min(rel_error2), jnp.min(rel_error3))
        vmax = max(jnp.max(rel_error1), jnp.max(rel_error2), jnp.max(rel_error3))
        # Create a TwoSlopeNorm object
        norm =TwoSlopeNorm(vmin=-(1e-12), vcenter=0, vmax=vmax)
        # Plot the relative errors
        plt.figure()
        plt.imshow(rel_error1, extent=[0, self.radius, self.t0, self.test_T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("PINN rel L2, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_rel_L2_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"PINN rel L2": wandb.Image(f"{save_path}/PINN_rel_L2_rho={rhomax}.png")} )
        print(f"PINN rel L2, rho={rhomax}->","min:",jnp.min(rel_error1),"max:",jnp.max(rel_error1),"mean:",jnp.mean(rel_error1))

        plt.figure()
        plt.imshow(rel_error2, extent=[0, self.radius, self.t0, self.test_T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("MLP rel L2, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/MLP_rel_L2_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"MLP rel L2": wandb.Image(f"{save_path}/MLP_rel_L2_rho={rhomax}.png")} )
        print(f"MLP rel L2, rho={rhomax}->","min:",jnp.min(rel_error2),"max:",jnp.max(rel_error2),"mean:",jnp.mean(rel_error2))

        plt.figure()
        plt.imshow(rel_error3, extent=[0, self.radius, self.t0, self.test_T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("ScaSML rel L2, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/ScaSML_rel_L2_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"ScaSML rel L2": wandb.Image(f"{save_path}/ScaSML_rel_L2_rho={rhomax}.png")} )
        print(f"ScaSML rel L2, rho={rhomax}->","min:",jnp.min(rel_error3),"max:",jnp.max(rel_error3),"mean:",jnp.mean(rel_error3))
            
        # Find the global minimum and maximum error
        vmin = min(jnp.min(errors1), jnp.min(errors2), jnp.min(errors3), jnp.min(errors_13), jnp.min(errors_23),jnp.min(errors_12),jnp.min(real_sol_abs))
        vmax = max(jnp.max(errors1), jnp.max(errors2), jnp.max(errors3), jnp.max(errors_13), jnp.max(errors_23),jnp.max(errors_12),jnp.max(real_sol_abs))
        # Create a TwoSlopeNorm object
        norm =TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        # Plot the real solution
        plt.figure()
        plt.imshow(real_sol_abs, extent=[0, self.radius, self.t0, self.test_T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("Real Solution")
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/Real_Solution.png")
        # Upload the plot to wandb
        wandb.log({"Real Solution": wandb.Image(f"{save_path}/Real_Solution.png")} )
        print("Real Solution->","min:",jnp.min(real_sol_abs),"max:",jnp.max(real_sol_abs),"mean:",jnp.mean(real_sol_abs))
        
        # Plot the errors
        plt.figure()
        plt.imshow(errors1, extent=[0, self.radius, self.t0, self.test_T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("PINN L2, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_L2_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"PINN L2": wandb.Image(f"{save_path}/PINN_L2_rho={rhomax}.png")} )
        print(f"PINN L2, rho={rhomax}->","min:",jnp.min(errors1),"max:",jnp.max(errors1),"mean:",jnp.mean(errors1))

        plt.figure()
        plt.imshow(errors2, extent=[0, self.radius, self.t0, self.test_T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("MLP L2, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/MLP_L2_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"MLP L2": wandb.Image(f"{save_path}/MLP_L2_rho={rhomax}.png")} )
        print(f"MLP L2, rho={rhomax}->","min:",jnp.min(errors2),"max:",jnp.max(errors2),"mean:",jnp.mean(errors2))

        plt.figure()
        plt.imshow(errors3, extent=[0, self.radius, self.t0, self.test_T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("ScaSML L2, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/ScaSML_L2_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"ScaSML L2": wandb.Image(f"{save_path}/ScaSML_L2_rho={rhomax}.png")} )
        print(f"ScaSML L2, rho={rhomax}->","min:",jnp.min(errors3),"max:",jnp.max(errors3),"mean:",jnp.mean(errors3))

        plt.figure()
        plt.imshow(errors_13, extent=[0, self.radius, self.t0, self.test_T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("PINN L2 - ScaSML L2, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_ScaSML_L2_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"PINN L2 - ScaSML L2": wandb.Image(f"{save_path}/PINN_ScaSML_L2_rho={rhomax}.png")} )

        plt.figure()
        plt.imshow(errors_23, extent=[0, self.radius, self.t0, self.test_T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("MLP L2 - ScaSML L2, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/MLP_ScaSML_L2_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"MLP L2 - ScaSML L2": wandb.Image(f"{save_path}/MLP_ScaSML_L2_rho={rhomax}.png")} )

        plt.figure()
        plt.imshow(errors_12, extent=[0, self.radius, self.t0, self.test_T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("PINN L2 - MLP L2, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_MLP_L2_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"PINN L2 - MLP L2": wandb.Image(f"{save_path}/PINN_MLP_L2_rho={rhomax}.png")} )

        # Calculate the sums of positive and negative differences
        positive_sum_13 = jnp.sum(errors_13[errors_13 > 0])
        negative_sum_13 = jnp.sum(errors_13[errors_13 < 0])
        positive_sum_23 = jnp.sum(errors_23[errors_23 > 0])
        negative_sum_23 = jnp.sum(errors_23[errors_23 < 0])
        postive_sum_12 = jnp.sum(errors_12[errors_12 > 0])
        negative_sum_12 = jnp.sum(errors_12[errors_12 < 0])
        # Display the positive count, negative count, positive sum, and negative sum of the difference of the errors
        print(f'PINN L2 - ScaSML L2,rho={rhomax}->','positve count:',jnp.sum(errors_13>0),'negative count:',jnp.sum(errors_13<0), 'positive sum:', positive_sum_13, 'negative sum:', negative_sum_13)
        print(f'MLP L2- ScaSML L2,rho={rhomax}->','positve count:',jnp.sum(errors_23>0),'negative count:',jnp.sum(errors_23<0), 'positive sum:', positive_sum_23, 'negative sum:', negative_sum_23)
        print(f'PINN L2 - MLP L2,rho={rhomax}->','positve count:',jnp.sum(errors_12>0),'negative count:',jnp.sum(errors_12<0), 'positive sum:', postive_sum_12, 'negative sum:', negative_sum_12)
        # Log the results to wandb
        wandb.log({f"mean of PINN L2,rho={rhomax}": jnp.mean(errors1), f"mean of MLP L2,rho={rhomax}": jnp.mean(errors2), f"mean of ScaSML L2,rho={rhomax}": jnp.mean(errors3)})
        wandb.log({f"min of PINN L2,rho={rhomax}": jnp.min(errors1), f"min of MLP L2,rho={rhomax}": jnp.min(errors2), f"min of ScaSML L2,rho={rhomax}": jnp.min(errors3)})
        wandb.log({f"max of PINN L2,rho={rhomax}": jnp.max(errors1), f"max of MLP L2,rho={rhomax}": jnp.max(errors2), f"max of ScaSML L2,rho={rhomax}": jnp.max(errors3)})
        wandb.log({f"positive count of PINN L2 - ScaSML L2,rho={rhomax}": jnp.sum(errors_13>0), f"negative count of PINN L2 - ScaSML L2,rho={rhomax}": jnp.sum(errors_13<0), f"positive sum of PINN L2 - ScaSML L2,rho={rhomax}": positive_sum_13, f"negative sum of PINN L2 - ScaSML L2,rho={rhomax}": negative_sum_13})
        wandb.log({f"positive count of MLP L2 - ScaSML L2,rho={rhomax}": jnp.sum(errors_23>0), f"negative count of MLP L2 - ScaSML L2,rho={rhomax}": jnp.sum(errors_23<0), f"positive sum of MLP L2 - ScaSML L2,rho={rhomax}": positive_sum_23, f"negative sum of MLP L2 - ScaSML L2,rho={rhomax}": negative_sum_23})
        wandb.log({f"positive count of PINN L2 - MLP L2,rho={rhomax}": jnp.sum(errors_12>0), f"negative count of PINN L2 - MLP L2,rho={rhomax}": jnp.sum(errors_12<0), f"positive sum of PINN L2 - MLP L2,rho={rhomax}": postive_sum_12, f"negative sum of PINN L2 - MLP L2,rho={rhomax}": negative_sum_12})
        # reset stdout and stderr
        sys.stdout=self.stdout
        sys.stderr=self.stderr
        #close the log file
        log_file.close()
        return rhomax

