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

class ConvergenceRate(object):
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

    def test(self, save_path, rhomax=4, n_samples=5000):
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
        eq_dim=eq.n_input-1
        geom=eq.geometry()
        evaluation_number_list1, evaluation_number_list2, evaluation_number_list3 = [], [], []  # evaluation_number_list1, evaluation_number_list2, evaluation_number_list3: list, initialized to empty list for timing each solver
        errors1_list, errors2_list, errors3_list = [], [], []  # errors1_list, errors2_list, errors3_list: list, initialized to empty list for storing errors
        for rho_ in range(2,rhomax+1):# at least two layers
            n=rho_
            self.solver2.set_approx_parameters(rho_)
            self.solver3.set_approx_parameters(rho_)
            errors1 = np.zeros(n_samples)  # errors1: ndarray, shape: (n_samples,), dtype: float
            errors2 = np.zeros(n_samples)  # errors2: ndarray, shape: (n_samples,), dtype: float
            errors3 = np.zeros(n_samples)  # errors3: ndarray, shape: (n_samples,), dtype: float
            evaluation_number1, evaluation_number2, evaluation_number3 = 0, 0, 0  # evaluation_number1, evaluation_number2, evaluation_number3: float, initialized to 0 for timing each solver

            # Compute the errors
            xt_values = geom.random_points(n_samples,random="Hammersley")  # xt_values: ndarray, shape: (n_samples, self.dim + 1), dtype: float
            exact_sol = eq.exact_solution(xt_values)  # exact_sol: ndarray, shape: (n_samples,), dtype: float

            # Measure the evaluation_number for solver1
            sol1 = self.solver1(torch.tensor(xt_values, dtype=torch.float32)).detach().cpu().numpy()[:, 0]  # sol1: ndarray, shape: (n_samples,), dtype: float
            evaluation_number1 += xt_values.shape[0]
            evaluation_number_list1.append(evaluation_number1)

            # Measure the evaluation_number for solver2
            sol2 = self.solver2.u_solve(n, rho_, xt_values)  # sol2: ndarray, shape: (n_samples,), dtype: float
            evaluation_number2 += self.solver2.evaluation_counter
            evaluation_number_list2.append(evaluation_number2)

            # Measure the evaluation_number for solver3
            sol3 = self.solver3.u_solve(n, rho_, xt_values)  # sol3: ndarray, shape: (n_samples,), dtype: float
            evaluation_number3 += self.solver3.evaluation_counter
            evaluation_number_list3.append(evaluation_number3)

            # Compute the errors
            errors1=np.abs(sol1 - exact_sol)
            errors2=np.abs(sol2 - exact_sol)
            errors3=np.abs(sol3 - exact_sol)

            # Compute the mean errors
            errors1_list.append(np.mean(errors1))
            errors2_list.append(np.mean(errors2))
            errors3_list.append(np.mean(errors3))
        
        epsilon = 1e-10
        # Convert lists to arrays
        evaluation_number_array1 = np.array(evaluation_number_list1)  # evaluation_number_array1: ndarray, shape: (rhomax,), dtype: float
        evaluation_number_array2 = np.array(evaluation_number_list2)  # evaluation_number_array2: ndarray, shape: (rhomax,), dtype: float
        evaluation_number_array3 = np.array(evaluation_number_list3)  # evaluation_number_array3: ndarray, shape: (rhomax,), dtype: float
        errors1_array = np.array(errors1_list)  # errors1_array: ndarray, shape: (rhomax,), dtype: float
        errors2_array = np.array(errors2_list)  # errors2_array: ndarray, shape: (rhomax,), dtype: float
        errors3_array = np.array(errors3_list)  # errors3_array: ndarray, shape: (rhomax,), dtype: float

        
        
        # Plot the convergence rate for PINN
        plt.figure()
        plt.plot(np.log10(evaluation_number_array1 + epsilon), np.log10(np.array(errors1_array) + epsilon), label='PINN')
        slope_1_2 = -1/2 * (np.log10(evaluation_number_array1+epsilon)-np.log10(evaluation_number_array1[0]+epsilon)) + np.log10(errors1_array[0] + epsilon)
        slope_1_4 = -1/4 * (np.log10(evaluation_number_array1+epsilon)-np.log10(evaluation_number_array1[0]+epsilon)) + np.log10(errors1_array[0] + epsilon)
        plt.plot(np.log10(evaluation_number_array1 + epsilon), slope_1_2, label='slope=-1/2')
        plt.plot(np.log10(evaluation_number_array1 + epsilon), slope_1_4, label='slope=-1/4')
        plt.scatter(np.log10(evaluation_number_array1 + epsilon), np.log10(np.array(errors1_array) + epsilon), marker='x')
        plt.scatter(np.log10(evaluation_number_array1 + epsilon), slope_1_2, marker='x')
        plt.scatter(np.log10(evaluation_number_array1 + epsilon), slope_1_4, marker='x')
        for i in range(len(evaluation_number_array1)):
            plt.annotate(i + 2, (np.log10(evaluation_number_array1[i] + epsilon), np.log10(np.array(errors1_array)[i] + epsilon)), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(i + 2, (np.log10(evaluation_number_array1[i] + epsilon), slope_1_2[i]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(i + 2, (np.log10(evaluation_number_array1[i] + epsilon), slope_1_4[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.title('PINN - Convergence Rate')
        plt.xlabel('log10(evaluation_number)')
        plt.ylabel('log10(error)')
        plt.legend()
        plt.savefig(f'{save_path}/PINN_convergence_rate.png')
        wandb.log({"PINN_convergence_rate": plt})
        
        # Plot the convergence rate for MLP
        plt.figure()
        plt.plot(np.log10(evaluation_number_array2 + epsilon), np.log10(np.array(errors2_array) + epsilon), label='MLP')
        slope_1_2_mlp = -1/2 * (np.log10(evaluation_number_array2+epsilon)-np.log10(evaluation_number_array2[0]+epsilon)) + np.log10(errors2_array[0] + epsilon)
        slope_1_4_mlp = -1/4 * (np.log10(evaluation_number_array2+epsilon)-np.log10(evaluation_number_array2[0]+epsilon)) + np.log10(errors2_array[0] + epsilon)
        plt.plot(np.log10(evaluation_number_array2 + epsilon), slope_1_2_mlp, label='MLP slope=-1/2')
        plt.plot(np.log10(evaluation_number_array2 + epsilon), slope_1_4_mlp, label='MLP slope=-1/4')
        plt.scatter(np.log10(evaluation_number_array2 + epsilon), np.log10(np.array(errors2_array) + epsilon), marker='x')
        plt.scatter(np.log10(evaluation_number_array2 + epsilon), slope_1_2_mlp, marker='x')
        plt.scatter(np.log10(evaluation_number_array2 + epsilon), slope_1_4_mlp, marker='x')
        for i in range(len(evaluation_number_array2)):
            plt.annotate(i + 2, (np.log10(evaluation_number_array2[i] + epsilon), np.log10(np.array(errors2_array)[i] + epsilon)), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(i + 2, (np.log10(evaluation_number_array2[i] + epsilon), slope_1_2_mlp[i]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(i + 2, (np.log10(evaluation_number_array2[i] + epsilon), slope_1_4_mlp[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.title('MLP - Convergence Rate')
        plt.xlabel('log10(evaluation_number)')
        plt.ylabel('log10(error)')
        plt.legend()
        plt.savefig(f'{save_path}/MLP_convergence_rate.png')
        wandb.log({"MLP_convergence_rate": plt})
        
        # Plot the convergence rate for ScaSML
        plt.figure()
        plt.plot(np.log10(evaluation_number_array3 + epsilon), np.log10(np.array(errors3_array) + epsilon), label='ScaSML')
        slope_1_2_scasml = -1/2 * (np.log10(evaluation_number_array3+epsilon)-np.log10(evaluation_number_array3[0]+epsilon)) + np.log10(errors3_array[0] + epsilon)
        slope_1_4_scasml = -1/4 * (np.log10(evaluation_number_array3+epsilon)-np.log10(evaluation_number_array3[0]+epsilon)) + np.log10(errors3_array[0] + epsilon)
        plt.plot(np.log10(evaluation_number_array3 + epsilon), slope_1_2_scasml, label='ScaSML slope=-1/2')
        plt.plot(np.log10(evaluation_number_array3 + epsilon), slope_1_4_scasml, label='ScaSML slope=-1/4')
        plt.scatter(np.log10(evaluation_number_array3 + epsilon), np.log10(np.array(errors3_array) + epsilon), marker='x')
        plt.scatter(np.log10(evaluation_number_array3 + epsilon), slope_1_2_scasml, marker='x')
        plt.scatter(np.log10(evaluation_number_array3 + epsilon), slope_1_4_scasml, marker='x')
        for i in range(len(evaluation_number_array3)):
            plt.annotate(i + 2, (np.log10(evaluation_number_array3[i] + epsilon), np.log10(np.array(errors3_array)[i] + epsilon)), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(i + 2, (np.log10(evaluation_number_array3[i] + epsilon), slope_1_2_scasml[i]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(i + 2, (np.log10(evaluation_number_array3[i] + epsilon), slope_1_4_scasml[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.title('ScaSML - Convergence Rate')
        plt.xlabel('log10(evaluation_number)')
        plt.ylabel('log10(error)')
        plt.legend()
        plt.savefig(f'{save_path}/ScaSML_convergence_rate.png')
        wandb.log({"ScaSML_convergence_rate": plt})
        
        # Plot MLP and ScaSML convergence rates on the same plot
        plt.figure()
        plt.plot(np.log10(evaluation_number_array2 + epsilon), np.log10(np.array(errors2_array) + epsilon), label='MLP')
        plt.plot(np.log10(evaluation_number_array3 + epsilon), np.log10(np.array(errors3_array) + epsilon), label='ScaSML')
        plt.plot(np.log10(evaluation_number_array2 + epsilon), slope_1_2_mlp, label='MLP slope=-1/2')
        plt.plot(np.log10(evaluation_number_array2 + epsilon), slope_1_4_mlp, label='MLP slope=-1/4')
        plt.plot(np.log10(evaluation_number_array3 + epsilon), slope_1_2_scasml, label='ScaSML slope=-1/2')
        plt.plot(np.log10(evaluation_number_array3 + epsilon), slope_1_4_scasml, label='ScaSML slope=-1/4')
        plt.scatter(np.log10(evaluation_number_array2 + epsilon), np.log10(np.array(errors2_array) + epsilon), marker='x')
        plt.scatter(np.log10(evaluation_number_array3 + epsilon), np.log10(np.array(errors3_array) + epsilon), marker='x')
        plt.scatter(np.log10(evaluation_number_array2 + epsilon), slope_1_2_mlp, marker='x')
        plt.scatter(np.log10(evaluation_number_array2 + epsilon), slope_1_4_mlp, marker='x')
        plt.scatter(np.log10(evaluation_number_array3 + epsilon), slope_1_2_scasml, marker='x')
        plt.scatter(np.log10(evaluation_number_array3 + epsilon), slope_1_4_scasml, marker='x')
        plt.title('MLP and ScaSML - Convergence Rate')
        plt.xlabel('log10(evaluation_number)')
        plt.ylabel('log10(error)')
        plt.legend()
        plt.savefig(f'{save_path}/MLP_ScaSML_convergence_rate.png')
        wandb.log({"MLP_ScaSML_convergence_rate": plt})
        
        return rhomax
