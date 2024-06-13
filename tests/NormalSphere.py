import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm
import time

class NormalSphere(object):
    '''Normal sphere test in high dimensions'''
    def __init__(self, equation, solver1,solver2,solver3):
        #initialize the normal spheres
        #solver1 for PINN network
        #solver2 for MLP object
        #solver3 for ScaML object
        self.equation=equation
        self.dim=equation.n_input-1
        solver1.eval()
        self.solver1=solver1
        self.solver2=solver2
        self.solver3=solver3
        self.t0=equation.t0
        self.T=equation.T
        self.radius=np.sqrt(self.dim*(self.T-self.t0)**2)
    def test(self,save_path,rhomax=2,n_samples=10,x_grid_num=100,t_grid_num=10):
        #compare solvers on different distances on the sphere
        eq=self.equation
        n=rhomax
        x_grid=np.linspace(0,self.radius,x_grid_num)
        t_grid=np.linspace(self.t0,self.T,t_grid_num)
        x_mesh,t_mesh=np.meshgrid(x_grid,t_grid)
        self.solver2.set_approx_parameters(rhomax)  
        self.solver3.set_approx_parameters(rhomax)  
        errors1=np.zeros_like(x_mesh)
        errors2=np.zeros_like(x_mesh)
        errors3=np.zeros_like(x_mesh)
        time1,time2,time3=0,0,0 
        # Compute the errors
        for i in tqdm(range(x_mesh.shape[0]), desc=f"Computing errors"):
            for j in tqdm(range(x_mesh.shape[1]), desc=f"Computing errors at time {t_grid[i]}"):
                x_values = np.random.normal(0, 1, (n_samples, self.dim))
                x_values /= np.linalg.norm(x_values, axis=1)[:, np.newaxis]
                x_values *= x_mesh[i, j]
                t_values = np.full((n_samples, 1), t_mesh[i, j])  # Create a 2D array filled with t_mesh[i, j]
                xt_values = np.concatenate((x_values, t_values), axis=1)
                exact_sol = eq.exact_solution(xt_values)

                # Measure the time for solver1
                start = time.time()
                sol1 = self.solver1(torch.tensor(xt_values, dtype=torch.float32)).detach().numpy()[:, 0]
                time1 += time.time() - start

                # Measure the time for solver2
                start = time.time()
                sol2 = self.solver2.u_solve(n, rhomax, xt_values)
                time2 += time.time() - start

                # Measure the time for solver3
                start = time.time()
                sol3 = self.solver3.u_solve(n, rhomax, xt_values)
                time3 += time.time() - start

                errors1[i, j] += np.mean(sol1 - exact_sol)
                errors2[i, j] += np.mean(sol2 - exact_sol)
                errors3[i, j] += np.mean(sol3 - exact_sol)

        # Print the total time for each solver
        print(f"Total time for PINN: {time1} seconds")
        print(f"Total time for MLP: {time2} seconds")
        print(f"Total time for ScaML: {time3} seconds")
        wandb.log({"Total time for PINN": time1, "Total time for MLP": time2, "Total time for ScaML": time3})
        # compute |errors1|-|errors3|,|errrors2|-|errors3|
        errors_13=np.abs(errors1)-np.abs(errors3)
        errors_23=np.abs(errors2)-np.abs(errors3)

        # Find the global minimum and maximum error
        vmin = min(np.min(errors1), np.min(errors2), np.min(errors3), np.min(errors_13), np.min(errors_23))
        vmax = max(np.max(errors1), np.max(errors2), np.max(errors3), np.max(errors_13), np.max(errors_23))
        # Create a TwoSlopeNorm object
        norm =TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        # Plot the errors
        plt.figure()
        plt.imshow(errors1, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("PINN error, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_error_rho={rhomax}.png")
        abs_errors1=np.abs(errors1)
        print(f"magnitude of PINN error, rho={rhomax}->","min:",np.min(abs_errors1),"max:",np.max(abs_errors1),"mean:",np.mean(abs_errors1))

        plt.figure()
        plt.imshow(errors2, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("MLP error, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/MLP_error_rho={rhomax}.png")
        abs_errors2=np.abs(errors2)
        print(f"maginitude of MLP error, rho={rhomax}->","min:",np.min(abs_errors2),"max:",np.max(abs_errors2),"mean:",np.mean(abs_errors2))

        plt.figure()
        plt.imshow(errors3, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("ScaML error, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/ScaML_error_rho={rhomax}.png")
        abs_errors3=np.abs(errors3)
        print(f"magnitude of ScaML error, rho={rhomax}->","min:",np.min(abs_errors3),"max:",np.max(abs_errors3),"mean:",np.mean(abs_errors3))

        plt.figure()
        plt.imshow(errors_13, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("|PINN error| - |ScaML error|, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_ScaML_error_rho={rhomax}.png")

        plt.figure()
        plt.imshow(errors_23, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("|MLP error| - |ScaML error|, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/MLP_ScaML_error_rho={rhomax}.png")
        # Calculate the sums of positive and negative differences
        positive_sum_13 = np.sum(errors_13[errors_13 > 0])
        negative_sum_13 = np.sum(errors_13[errors_13 < 0])
        positive_sum_23 = np.sum(errors_23[errors_23 > 0])
        negative_sum_23 = np.sum(errors_23[errors_23 < 0])
        # Display the positive count, negative count, positive sum, and negative sum of the difference of the errors
        print(f'|PINN error| - |ScaML error|,rho={rhomax}->','positve count:',np.sum(errors_13>0),'negative count:',np.sum(errors_13<0), 'positive sum:', positive_sum_13, 'negative sum:', negative_sum_13)
        print(f'|MLP error| - |ScaML error|,rho={rhomax}->','positve count:',np.sum(errors_23>0),'negative count:',np.sum(errors_23<0), 'positive sum:', positive_sum_23, 'negative sum:', negative_sum_23)
        # Log the results to wandb
        wandb.log({f"PINN error,rho={rhomax}": wandb.Image(f"{save_path}/PINN_error_rho={rhomax}.png"), f"MLP error,rho={rhomax}": wandb.Image(f"{save_path}/MLP_error_rho={rhomax}.png"), f"ScaML error,rho={rhomax}": wandb.Image(f"{save_path}/ScaML_error_rho={rhomax}.png")})
        wandb.log({f"PINN-ScaML error,rho={rhomax}": wandb.Image(f"{save_path}/PINN_ScaML_error_rho={rhomax}.png"), f"MLP-ScaML error,rho={rhomax}": wandb.Image(f"{save_path}/MLP_ScaML_error_rho={rhomax}.png")})
        wandb.log({f"mean magnitude of PINN error,rho={rhomax}": np.mean(abs_errors1), f"mean magnitude of MLP error,rho={rhomax}": np.mean(abs_errors2), f"mean magnitude of ScaML error,rho={rhomax}": np.mean(abs_errors3)})
        wandb.log({f"max maginitude of PINN error,rho={rhomax}": np.max(abs_errors1), f"max maginitude of MLP error,rho={rhomax}": np.max(abs_errors2), f"max maginitude of ScaML error,rho={rhomax}": np.max(abs_errors3)})
        wandb.log({f"min maginitude of PINN error,rho={rhomax}": np.min(abs_errors1), f"min maginitude of MLP error,rho={rhomax}": np.min(abs_errors2), f"min maginitude of ScaML error,rho={rhomax}": np.min(abs_errors3)})
        wandb.log({f"|PINN error| - |ScaML error|,rho={rhomax}-> positve count": np.sum(errors_13>0), f"|PINN error| - |ScaML error|,rho={rhomax}-> negative count": np.sum(errors_13<0)}) 
        wandb.log({f"|PINN error| - |ScaML error|,rho={rhomax}-> positive sum": positive_sum_13, f"|PINN error| - |ScaML error|,rho={rhomax}-> negative sum": negative_sum_13})
        wandb.log({f"|MLP error| - |ScaML error|,rho={rhomax}-> positve count": np.sum(errors_23>0), f"|MLP error| - |ScaML error|,rho={rhomax}-> negative count": np.sum(errors_23<0)})
        wandb.log({f"|MLP error| - |ScaML error|,rho={rhomax}-> positive sum": positive_sum_23, f"|MLP error| - |ScaML error|,rho={rhomax}-> negative sum": negative_sum_23})
        return rhomax

