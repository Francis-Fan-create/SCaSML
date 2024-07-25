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

class NormalSphere(object):
    '''
    Normal sphere test in high dimensions.

    Attributes:
    equation (object): An object representing the equation to solve.
    dim (int): The dimension of the input space minus one.
    solver1 (object): A PyTorch model for the PINN network.
    solver2 (object): An object for the MLP solver.
    solver3 (object): An object for the ScaSML solver.
    t0 (float): The initial time.
    T (float): The final time.
    radius (float): The radius of the sphere calculated based on the dimension and time.
    '''
    def __init__(self, equation, solver1, solver2, solver3):
        '''
        Initializes the normal spheres with given solvers and equation.

        Parameters:
        equation (object): The equation object containing problem specifics.
        solver1 (object): The PINN network solver.
        solver2 (object): The MLP solver object.
        solver3 (object): The ScaSML solver object.
        '''
        #save original stdout and stderr
        self.stdout=sys.stdout
        self.stderr=sys.stderr
        # Initialize the normal spheres
        self.equation = equation
        self.dim = equation.n_input - 1  # equation.n_input: int
        solver1.eval()  # Set the PINN network to evaluation mode
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.t0 = equation.t0  # equation.t0: float
        self.T = equation.T  # equation.T: float
        self.radius = np.sqrt(self.dim * (self.T - self.t0) ** 2)  # radius: float, calculated based on dimension and time

    def test(self, save_path, rhomax=2, n_samples=50, x_grid_num=100, t_grid_num=10):
        '''
        Compares solvers on different distances on the sphere.

        Parameters:
        save_path (str): The path to save the results.
        rhomax (int): The maximum value of rho for approximation parameters.
        n_samples (int): The number of samples for testing.
        x_grid_num (int): The number of grid points in the x dimension.
        t_grid_num (int): The number of grid points in the time dimension.
        '''
        #initialize the profiler
        profiler = cProfile.Profile()
        profiler.enable()
        # create the save path if it does not exist
        class_name = self.__class__.__name__
        new_path = f"{save_path}/{class_name}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        save_path = new_path
        directory=f'{save_path}/callbacks'
        # Delete former callbacks
        if os.path.exists(directory):
            # iterate over the files in the directory
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    # if it is a file or a link, delete it
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    # if it is a directory, delete it
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')        
        eq = self.equation
        eq_name = eq.__class__.__name__
        n=rhomax
        x_grid = np.linspace(0, self.radius, x_grid_num)  # x_grid: ndarray, shape: (x_grid_num,), dtype: float
        t_grid = np.linspace(self.t0, self.T, t_grid_num)  # t_grid: ndarray, shape: (t_grid_num,), dtype: float
        x_mesh, t_mesh = np.meshgrid(x_grid, t_grid)  # x_mesh, t_mesh: ndarray, shape: (t_grid_num, x_grid_num), dtype: float
        self.solver2.set_approx_parameters(rhomax)
        self.solver3.set_approx_parameters(rhomax)
        errors1 = np.zeros_like(x_mesh)  # errors1: ndarray, shape: (t_grid_num, x_grid_num), dtype: float
        errors2 = np.zeros_like(x_mesh)  # errors2: ndarray, shape: (t_grid_num, x_grid_num), dtype: float
        errors3 = np.zeros_like(x_mesh)  # errors3: ndarray, shape: (t_grid_num, x_grid_num), dtype: float
        rel_error1 = np.zeros_like(x_mesh)  # rel_error1: ndarray, shape: (t_grid_num, x_grid_num), dtype: float
        rel_error2 = np.zeros_like(x_mesh)  # rel_error2: ndarray, shape: (t_grid_num, x_grid_num), dtype: float
        rel_error3 = np.zeros_like(x_mesh)  # rel_error3: ndarray, shape: (t_grid_num, x_grid_num), dtype: float
        real_sol_abs = np.zeros_like(x_mesh)  # real_sol_abs: ndarray, shape: (t_grid_num, x_grid_num), dtype: float
        time1, time2, time3 = 0, 0, 0  # time1, time2, time3: float, initialized to 0 for timing each solver

        # Compute the errors
        for i in tqdm(range(x_mesh.shape[0]), desc=f"Computing errors"):
            for j in tqdm(range(x_mesh.shape[1]), desc=f"Computing errors at time {t_grid[i]}"):
                x_values = np.random.normal(0, 1, (n_samples, self.dim))  # x_values: ndarray, shape: (n_samples, self.dim), dtype: float
                x_values /= np.linalg.norm(x_values, axis=1)[:, np.newaxis]  # Normalize x_values
                x_values *= x_mesh[i, j]  # Scale x_values by x_mesh[i, j]
                t_values = np.full((n_samples, 1), t_mesh[i, j])  # t_values: ndarray, shape: (n_samples, 1), dtype: float
                xt_values = np.concatenate((x_values, t_values), axis=1)  # xt_values: ndarray, shape: (n_samples, self.dim + 1), dtype: float
                exact_sol = eq.exact_solution(xt_values)  # exact_sol: ndarray, shape: (n_samples,), dtype: float

                # Measure the time for solver1
                start = time.time()
                sol1 = self.solver1(torch.tensor(xt_values, dtype=torch.float32)).detach().cpu().numpy()[:, 0]  # sol1: ndarray, shape: (n_samples,), dtype: float
                time1 += time.time() - start

                # Measure the time for solver2
                start = time.time()
                sol2 = self.solver2.u_solve(n, rhomax, xt_values)  # sol2: ndarray, shape: (n_samples,), dtype: float
                time2 += time.time() - start

                # Measure the time for solver3
                start = time.time()
                sol3 = self.solver3.u_solve(n, rhomax, xt_values)  # sol3: ndarray, shape: (n_samples,), dtype: float
                time3 += time.time() - start

                # # Compute the average error and relative error
                # errors1[i, j] += np.mean(np.abs(sol1 - exact_sol))
                # errors2[i, j] += np.mean(np.abs(sol2 - exact_sol))
                # errors3[i, j] += np.mean(np.abs(sol3 - exact_sol))
                # rel_error1[i, j] += np.mean(np.abs(sol1 - exact_sol) / (np.abs(exact_sol)+1e-6))
                # rel_error2[i, j] += np.mean(np.abs(sol2 - exact_sol) / (np.abs(exact_sol)+1e-6))
                # rel_error3[i, j] += np.mean(np.abs(sol3 - exact_sol) / (np.abs(exact_sol)+1e-6))
                # # Compute the average absolute value of the real solution
                # real_sol_abs[i, j] = np.mean(np.abs(exact_sol))  
                # Compute the maximum error and relative error
                errors1[i, j] = np.max(np.abs(sol1 - exact_sol))
                errors2[i, j] = np.max(np.abs(sol2 - exact_sol))
                errors3[i, j] = np.max(np.abs(sol3 - exact_sol))
                rel_error1[i, j] = np.max(np.abs(sol1 - exact_sol) / (np.abs(exact_sol)+1e-6))
                rel_error2[i, j] = np.max(np.abs(sol2 - exact_sol) / (np.abs(exact_sol)+1e-6))
                rel_error3[i, j] = np.max(np.abs(sol3 - exact_sol) / (np.abs(exact_sol)+1e-6))
                # Compute the maximum absolute value of the real solution
                real_sol_abs[i, j] = np.max(np.abs(exact_sol))

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
        errors = [errors1.flatten(), errors2.flatten(), errors3.flatten(), errors_13.flatten(), errors_23.flatten()]
        errors= [errors1.flatten(), errors2.flatten(), errors3.flatten()]
        # Create a boxplot
        # plt.boxplot(errors, labels=['PINN_l1', 'MLP_l1', 'ScaSML_l1', 'PINN_l1 - ScaSML_l1', 'MLP_l1 - ScaSML_l1'])
        plt.boxplot(errors, labels=['PINN_l1', 'MLP_l1', 'ScaSML_l1'])
        plt.xticks(rotation=45)
        # Add a title and labels
        plt.title('Absolute Error Distribution')
        plt.ylabel('Absolute Error Value')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f"{save_path}/Absolute_Error_Distribution.png")
        # Upload the plot to wandb
        wandb.log({"Error Distribution": wandb.Image(f"{save_path}/Absolute_Error_Distribution.png")})

        plt.figure()
        # collect all absolute errors
        errors = [errors1.flatten(), errors2.flatten(), errors3.flatten()]
        # Calculate means and standard deviations
        means = [np.mean(e) for e in errors]
        stds = [np.std(e) for e in errors]
        # Define labels
        labels = ['PINN_l1', 'MLP_l1', 'ScaSML_l1']
        x_pos = range(len(labels))
        # Create an error bar plot
        plt.errorbar(x_pos, means, yerr=stds, capsize=5, capthick=2, ecolor='black',  marker='s', markersize=7, mfc='red', mec='black')
        plt.xticks(x_pos, labels, rotation=45)
        # Add a title and labels
        plt.title('Absolute Error Distribution')
        plt.ylabel('Absolute Error Value')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f"{save_path}/Absolute_Error_Distribution_errorbar.png")
        # Upload the plot to wandb
        wandb.log({"Error Distribution": wandb.Image(f"{save_path}/Absolute_Error_Distribution_errorbar.png")})

        plt.figure()
        #collect all relative errors
        rel_errors = [rel_error1.flatten(), rel_error2.flatten(), rel_error3.flatten()]
        # Create a boxplot
        plt.boxplot(rel_errors, labels=['PINN_l1', 'MLP_l1', 'ScaSML_l1'])
        plt.xticks(rotation=45)
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
        means = [np.mean(errors) for errors in rel_errors]
        stds = [np.std(errors) for errors in rel_errors]
        # Define labels for each group
        labels = ['PINN_l1', 'MLP_l1', 'ScaSML_l1']
        x_pos = range(len(labels))
        # Create an error bar plot
        plt.errorbar(x_pos, means, yerr=stds, capsize=5, capthick=2, ecolor='black',  marker='s', markersize=7, mfc='red', mec='black')
        # Set the x-ticks to use the labels and rotate them for better readability
        plt.xticks(x_pos, labels, rotation=45)
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
        vmin = min(np.min(rel_error1), np.min(rel_error2), np.min(rel_error3))
        vmax = max(np.max(rel_error1), np.max(rel_error2), np.max(rel_error3))
        # Create a TwoSlopeNorm object
        norm =TwoSlopeNorm(vmin=-(1e-12), vcenter=0, vmax=vmax)
        # Plot the relative errors
        plt.figure()
        plt.imshow(rel_error1, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("PINN rel l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_rel_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"PINN rel l1": wandb.Image(f"{save_path}/PINN_rel_l1_rho={rhomax}.png")} )
        print(f"PINN rel l1, rho={rhomax}->","min:",np.min(rel_error1),"max:",np.max(rel_error1),"mean:",np.mean(rel_error1))

        plt.figure()
        plt.imshow(rel_error2, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("MLP rel l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/MLP_rel_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"MLP rel l1": wandb.Image(f"{save_path}/MLP_rel_l1_rho={rhomax}.png")} )
        print(f"MLP rel l1, rho={rhomax}->","min:",np.min(rel_error2),"max:",np.max(rel_error2),"mean:",np.mean(rel_error2))

        plt.figure()
        plt.imshow(rel_error3, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("ScaSML rel l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/ScaSML_rel_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"ScaSML rel l1": wandb.Image(f"{save_path}/ScaSML_rel_l1_rho={rhomax}.png")} )
        print(f"ScaSML rel l1, rho={rhomax}->","min:",np.min(rel_error3),"max:",np.max(rel_error3),"mean:",np.mean(rel_error3))
         
        # Find the global minimum and maximum error
        vmin = min(np.min(errors1), np.min(errors2), np.min(errors3), np.min(errors_13), np.min(errors_23),np.min(errors_12),np.min(real_sol_abs))
        vmax = max(np.max(errors1), np.max(errors2), np.max(errors3), np.max(errors_13), np.max(errors_23),np.max(errors_12),np.max(real_sol_abs))
        # Create a TwoSlopeNorm object
        norm =TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        # Plot the real solution
        plt.figure()
        plt.imshow(real_sol_abs, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("Real Solution")
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/Real_Solution.png")
        # Upload the plot to wandb
        wandb.log({"Real Solution": wandb.Image(f"{save_path}/Real_Solution.png")} )
        print("Real Solution->","min:",np.min(real_sol_abs),"max:",np.max(real_sol_abs),"mean:",np.mean(real_sol_abs))
        
        # Plot the errors
        plt.figure()
        plt.imshow(errors1, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("PINN l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"PINN l1": wandb.Image(f"{save_path}/PINN_l1_rho={rhomax}.png")} )
        print(f"PINN l1, rho={rhomax}->","min:",np.min(errors1),"max:",np.max(errors1),"mean:",np.mean(errors1))

        plt.figure()
        plt.imshow(errors2, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("MLP l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/MLP_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"MLP l1": wandb.Image(f"{save_path}/MLP_l1_rho={rhomax}.png")} )
        print(f"MLP l1, rho={rhomax}->","min:",np.min(errors2),"max:",np.max(errors2),"mean:",np.mean(errors2))

        plt.figure()
        plt.imshow(errors3, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("ScaSML l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/ScaSML_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"ScaSML l1": wandb.Image(f"{save_path}/ScaSML_l1_rho={rhomax}.png")} )
        print(f"ScaSML l1, rho={rhomax}->","min:",np.min(errors3),"max:",np.max(errors3),"mean:",np.mean(errors3))

        plt.figure()
        plt.imshow(errors_13, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("PINN l1 - ScaSML l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_ScaSML_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"PINN l1 - ScaSML l1": wandb.Image(f"{save_path}/PINN_ScaSML_l1_rho={rhomax}.png")} )

        plt.figure()
        plt.imshow(errors_23, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("MLP l1 - ScaSML l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/MLP_ScaSML_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"MLP l1 - ScaSML l1": wandb.Image(f"{save_path}/MLP_ScaSML_l1_rho={rhomax}.png")} )

        plt.figure()
        plt.imshow(errors_12, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("PINN l1 - MLP l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_MLP_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"PINN l1 - MLP l1": wandb.Image(f"{save_path}/PINN_MLP_l1_rho={rhomax}.png")} )

        # Calculate the sums of positive and negative differences
        positive_sum_13 = np.sum(errors_13[errors_13 > 0])
        negative_sum_13 = np.sum(errors_13[errors_13 < 0])
        positive_sum_23 = np.sum(errors_23[errors_23 > 0])
        negative_sum_23 = np.sum(errors_23[errors_23 < 0])
        postive_sum_12 = np.sum(errors_12[errors_12 > 0])
        negative_sum_12 = np.sum(errors_12[errors_12 < 0])
        # Display the positive count, negative count, positive sum, and negative sum of the difference of the errors
        print(f'PINN l1 - ScaSML l1,rho={rhomax}->','positve count:',np.sum(errors_13>0),'negative count:',np.sum(errors_13<0), 'positive sum:', positive_sum_13, 'negative sum:', negative_sum_13)
        print(f'MLP l1- ScaSML l1,rho={rhomax}->','positve count:',np.sum(errors_23>0),'negative count:',np.sum(errors_23<0), 'positive sum:', positive_sum_23, 'negative sum:', negative_sum_23)
        print(f'PINN l1 - MLP l1,rho={rhomax}->','positve count:',np.sum(errors_12>0),'negative count:',np.sum(errors_12<0), 'positive sum:', postive_sum_12, 'negative sum:', negative_sum_12)
        # Log the results to wandb
        wandb.log({f"mean of PINN l1,rho={rhomax}": np.mean(errors1), f"mean of MLP l1,rho={rhomax}": np.mean(errors2), f"mean of ScaSML l1,rho={rhomax}": np.mean(errors3)})
        wandb.log({f"min of PINN l1,rho={rhomax}": np.min(errors1), f"min of MLP l1,rho={rhomax}": np.min(errors2), f"min of ScaSML l1,rho={rhomax}": np.min(errors3)})
        wandb.log({f"max of PINN l1,rho={rhomax}": np.max(errors1), f"max of MLP l1,rho={rhomax}": np.max(errors2), f"max of ScaSML l1,rho={rhomax}": np.max(errors3)})
        wandb.log({f"positive count of PINN l1 - ScaSML l1,rho={rhomax}": np.sum(errors_13>0), f"negative count of PINN l1 - ScaSML l1,rho={rhomax}": np.sum(errors_13<0), f"positive sum of PINN l1 - ScaSML l1,rho={rhomax}": positive_sum_13, f"negative sum of PINN l1 - ScaSML l1,rho={rhomax}": negative_sum_13})
        wandb.log({f"positive count of MLP l1 - ScaSML l1,rho={rhomax}": np.sum(errors_23>0), f"negative count of MLP l1 - ScaSML l1,rho={rhomax}": np.sum(errors_23<0), f"positive sum of MLP l1 - ScaSML l1,rho={rhomax}": positive_sum_23, f"negative sum of MLP l1 - ScaSML l1,rho={rhomax}": negative_sum_23})
        wandb.log({f"positive count of PINN l1 - MLP l1,rho={rhomax}": np.sum(errors_12>0), f"negative count of PINN l1 - MLP l1,rho={rhomax}": np.sum(errors_12<0), f"positive sum of PINN l1 - MLP l1,rho={rhomax}": postive_sum_12, f"negative sum of PINN l1 - MLP l1,rho={rhomax}": negative_sum_12})
        # reset stdout and stderr
        sys.stdout=self.stdout
        sys.stderr=self.stderr
        #close the log file
        log_file.close()
        return rhomax

