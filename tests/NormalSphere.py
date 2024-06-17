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
                sol1 = self.solver1(torch.tensor(xt_values, dtype=torch.float32)).detach().cpu().numpy()[:, 0]
                time1 += time.time() - start

                # Measure the time for solver2
                start = time.time()
                sol2 = self.solver2.u_solve(n, rhomax, xt_values)
                time2 += time.time() - start

                # Measure the time for solver3
                start = time.time()
                sol3 = self.solver3.u_solve(n, rhomax, xt_values)
                time3 += time.time() - start
                # Compute the average error
                errors1[i, j] += np.mean(sol1 - exact_sol)
                errors2[i, j] += np.mean(sol2 - exact_sol)
                errors3[i, j] += np.mean(sol3 - exact_sol)
        errors1=np.abs(errors1)
        errors2=np.abs(errors2)
        errors3=np.abs(errors3)
        # Print the total time for each solver
        print(f"Total time for PINN: {time1} seconds")
        print(f"Total time for MLP: {time2} seconds")
        print(f"Total time for ScaML: {time3} seconds")
        wandb.log({"Total time for PINN": time1, "Total time for MLP": time2, "Total time for ScaML": time3})
        # compute |errors1|-|errors3|,|errrors2|-|errors3|
        errors_13=errors1-errors3
        errors_23=errors2-errors3
        
        # collect all errors
        errors = [errors1.flatten(), errors2.flatten(), errors3.flatten(), errors_13.flatten(), errors_23.flatten()]
        # Create a boxplot
        plt.boxplot(errors, labels=['PINN_l1', 'MLP_l1', 'ScaML_l1', 'PINN_l1 - ScaML_l1', 'MLP_l1 - ScaML_l1'])
        plt.xticks(rotation=45)
        # Add a title and labels
        plt.title('Error Distribution')
        plt.ylabel('Error Value')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f"{save_path}/Error_Distribution.png")
        # Upload the plot to wandb
        wandb.log({"Error Distribution": wandb.Image(f"{save_path}/Error_Distribution.png")})

        # Find the global minimum and maximum error
        vmin = min(np.min(errors1), np.min(errors2), np.min(errors3), np.min(errors_13), np.min(errors_23))
        vmax = max(np.max(errors1), np.max(errors2), np.max(errors3), np.max(errors_13), np.max(errors_23))
        # Create a TwoSlopeNorm object
        norm =TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
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
        plt.title("ScaML l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/ScaML_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"ScaML l1": wandb.Image(f"{save_path}/ScaML_l1_rho={rhomax}.png")} )
        print(f"ScaML l1, rho={rhomax}->","min:",np.min(errors3),"max:",np.max(errors3),"mean:",np.mean(errors3))

        plt.figure()
        plt.imshow(errors_13, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("PINN l1 - ScaML l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_ScaML_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"PINN l1 - ScaML l1": wandb.Image(f"{save_path}/PINN_ScaML_l1_rho={rhomax}.png")} )

        plt.figure()
        plt.imshow(errors_23, extent=[0, self.radius, self.t0, self.T], aspect='auto', cmap='RdBu_r',norm=norm)
        plt.colorbar()
        plt.title("MLP l1 - ScaML l1, rho={:d}".format(rhomax))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/MLP_ScaML_l1_rho={rhomax}.png")
        # Upload the plot to wandb
        wandb.log({"MLP l1 - ScaML l1": wandb.Image(f"{save_path}/MLP_ScaML_l1_rho={rhomax}.png")} )
        # Calculate the sums of positive and negative differences
        positive_sum_13 = np.sum(errors_13[errors_13 > 0])
        negative_sum_13 = np.sum(errors_13[errors_13 < 0])
        positive_sum_23 = np.sum(errors_23[errors_23 > 0])
        negative_sum_23 = np.sum(errors_23[errors_23 < 0])
        # Display the positive count, negative count, positive sum, and negative sum of the difference of the errors
        print(f'PINN l1 - ScaML l1,rho={rhomax}->','positve count:',np.sum(errors_13>0),'negative count:',np.sum(errors_13<0), 'positive sum:', positive_sum_13, 'negative sum:', negative_sum_13)
        print(f'MLP l1- ScaML l1,rho={rhomax}->','positve count:',np.sum(errors_23>0),'negative count:',np.sum(errors_23<0), 'positive sum:', positive_sum_23, 'negative sum:', negative_sum_23)
        # Log the results to wandb
        wandb.log({f"mean of PINN l1,rho={rhomax}": np.mean(errors1), f"mean of MLP l1,rho={rhomax}": np.mean(errors2), f"mean of ScaML l1,rho={rhomax}": np.mean(errors3)})
        wandb.log({f"min of PINN l1,rho={rhomax}": np.min(errors1), f"min of MLP l1,rho={rhomax}": np.min(errors2), f"min of ScaML l1,rho={rhomax}": np.min(errors3)})
        wandb.log({f"max of PINN l1,rho={rhomax}": np.max(errors1), f"max of MLP l1,rho={rhomax}": np.max(errors2), f"max of ScaML l1,rho={rhomax}": np.max(errors3)})
        wandb.log({f"positive count of PINN l1 - ScaML l1,rho={rhomax}": np.sum(errors_13>0), f"negative count of PINN l1 - ScaML l1,rho={rhomax}": np.sum(errors_13<0), f"positive sum of PINN l1 - ScaML l1,rho={rhomax}": positive_sum_13, f"negative sum of PINN l1 - ScaML l1,rho={rhomax}": negative_sum_13})
        wandb.log({f"positive count of MLP l1 - ScaML l1,rho={rhomax}": np.sum(errors_23>0), f"negative count of MLP l1 - ScaML l1,rho={rhomax}": np.sum(errors_23<0), f"positive sum of MLP l1 - ScaML l1,rho={rhomax}": positive_sum_23, f"negative sum of MLP l1 - ScaML l1,rho={rhomax}": negative_sum_23})
        return rhomax

