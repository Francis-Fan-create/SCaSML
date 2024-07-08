import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import time
import sys
import os
import cProfile
import shutil

class SimpleUniform(object):
    '''
    Simple Uniform test in high dimensions.

    Attributes:
    equation (object): An object representing the equation to solve.
    dim (int): The dimension of the input space minus one.
    solver1 (object): A PyTorch model for the PINN network.
    solver2 (object): An object for the MLP solver.
    solver3 (object): An object for the ScaSML solver.
    t0 (float): The initial time.
    T (float): The final time.
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

    def test(self, save_path, rhomax=2, n_samples=50):
        '''
        Compares solvers on different distances on the sphere.

        Parameters:
        save_path (str): The path to save the results.
        rhomax (int): The maximum value of rho for approximation parameters.
        n_samples (int): The number of samples for testing.

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
        # Set the approximation parameters
        eq = self.equation
        eq_name = eq.__class__.__name__
        geom=eq.geometry()
        n=rhomax
        self.solver2.set_approx_parameters(rhomax)
        self.solver3.set_approx_parameters(rhomax)
        errors1 = np.zeros(n_samples)  # errors1: ndarray, shape: (n_samples,), dtype: float
        errors2 = np.zeros(n_samples)  # errors2: ndarray, shape: (n_samples,), dtype: float
        errors3 = np.zeros(n_samples)  # errors3: ndarray, shape: (n_samples,), dtype: float
        rel_error1 = np.zeros(n_samples)  # rel_error1: ndarray, shape: (n_samples,), dtype: float
        rel_error2 = np.zeros(n_samples)  # rel_error2: ndarray, shape: (n_samples,), dtype: float
        rel_error3 = np.zeros(n_samples)  # rel_error3: ndarray, shape: (n_samples,), dtype: float
        real_sol_abs = np.zeros(n_samples)  # real_sol_abs: ndarray, shape: (n_samples,), dtype: float
        time1, time2, time3 = 0, 0, 0  # time1, time2, time3: float, initialized to 0 for timing each solver

        # Compute the errors
        xt_values = geom.random_points(n_samples,random="Hammersley")  # xt_values: ndarray, shape: (n_samples, self.dim + 1), dtype: float
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

        # Compute the average error and relative error
        errors1=np.abs(sol1 - exact_sol)
        errors2=np.abs(sol2 - exact_sol)
        errors3=np.abs(sol3 - exact_sol)
        rel_error1= np.abs(sol1 - exact_sol) / (np.abs(exact_sol)+1e-6)
        rel_error2= np.abs(sol2 - exact_sol) / (np.abs(exact_sol)+1e-6)
        rel_error3= np.abs(sol3 - exact_sol) / (np.abs(exact_sol)+1e-6)
        real_sol_abs= np.abs(exact_sol)  # Compute the absolute value of the real solution
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
        errors_13=errors1-errors3
        errors_23=errors2-errors3
        
        plt.figure()
        # collect all absolute errors
        # errors = [errors1.flatten(), errors2.flatten(), errors3.flatten(), errors_13.flatten(), errors_23.flatten()]
        errors = [errors1.flatten(), errors2.flatten(), errors3.flatten()]
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

        
        print(f"PINN rel l1, rho={rhomax}->","min:",np.min(rel_error1),"max:",np.max(rel_error1),"mean:",np.mean(rel_error1))

        
        print(f"MLP rel l1, rho={rhomax}->","min:",np.min(rel_error2),"max:",np.max(rel_error2),"mean:",np.mean(rel_error2))

       
        print(f"ScaSML rel l1, rho={rhomax}->","min:",np.min(rel_error3),"max:",np.max(rel_error3),"mean:",np.mean(rel_error3))
         
        
        print("Real Solution->","min:",np.min(real_sol_abs),"max:",np.max(real_sol_abs),"mean:",np.mean(real_sol_abs))
        
       
        print(f"PINN l1, rho={rhomax}->","min:",np.min(errors1),"max:",np.max(errors1),"mean:",np.mean(errors1))

        
        print(f"MLP l1, rho={rhomax}->","min:",np.min(errors2),"max:",np.max(errors2),"mean:",np.mean(errors2))

       
        print(f"ScaSML l1, rho={rhomax}->","min:",np.min(errors3),"max:",np.max(errors3),"mean:",np.mean(errors3))


        # Calculate the sums of positive and negative differences
        positive_sum_13 = np.sum(errors_13[errors_13 > 0])
        negative_sum_13 = np.sum(errors_13[errors_13 < 0])
        positive_sum_23 = np.sum(errors_23[errors_23 > 0])
        negative_sum_23 = np.sum(errors_23[errors_23 < 0])
        # Display the positive count, negative count, positive sum, and negative sum of the difference of the errors
        print(f'PINN l1 - ScaSML l1,rho={rhomax}->','positve count:',np.sum(errors_13>0),'negative count:',np.sum(errors_13<0), 'positive sum:', positive_sum_13, 'negative sum:', negative_sum_13)
        print(f'MLP l1- ScaSML l1,rho={rhomax}->','positve count:',np.sum(errors_23>0),'negative count:',np.sum(errors_23<0), 'positive sum:', positive_sum_23, 'negative sum:', negative_sum_23)
        # Log the results to wandb
        wandb.log({f"mean of PINN l1,rho={rhomax}": np.mean(errors1), f"mean of MLP l1,rho={rhomax}": np.mean(errors2), f"mean of ScaSML l1,rho={rhomax}": np.mean(errors3)})
        wandb.log({f"min of PINN l1,rho={rhomax}": np.min(errors1), f"min of MLP l1,rho={rhomax}": np.min(errors2), f"min of ScaSML l1,rho={rhomax}": np.min(errors3)})
        wandb.log({f"max of PINN l1,rho={rhomax}": np.max(errors1), f"max of MLP l1,rho={rhomax}": np.max(errors2), f"max of ScaSML l1,rho={rhomax}": np.max(errors3)})
        wandb.log({f"positive count of PINN l1 - ScaSML l1,rho={rhomax}": np.sum(errors_13>0), f"negative count of PINN l1 - ScaSML l1,rho={rhomax}": np.sum(errors_13<0), f"positive sum of PINN l1 - ScaSML l1,rho={rhomax}": positive_sum_13, f"negative sum of PINN l1 - ScaSML l1,rho={rhomax}": negative_sum_13})
        wandb.log({f"positive count of MLP l1 - ScaSML l1,rho={rhomax}": np.sum(errors_23>0), f"negative count of MLP l1 - ScaSML l1,rho={rhomax}": np.sum(errors_23<0), f"positive sum of MLP l1 - ScaSML l1,rho={rhomax}": positive_sum_23, f"negative sum of MLP l1 - ScaSML l1,rho={rhomax}": negative_sum_23})
        # reset stdout and stderr
        sys.stdout=self.stdout
        sys.stderr=self.stderr
        #close the log file
        log_file.close()
        return rhomax
