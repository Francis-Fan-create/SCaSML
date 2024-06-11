import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
from tqdm import tqdm
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
    def test(self,save_path,rho=2,n_samples=10,x_grid_num=100,t_grid_num=10):
        #compare solvers on different distances on the sphere
        eq=self.equation
        n=rho
        x_grid=np.linspace(0,self.radius,x_grid_num)
        t_grid=np.linspace(self.t0,self.T,t_grid_num)
        x_mesh,t_mesh=np.meshgrid(x_grid,t_grid)
        errors1=np.zeros_like(x_mesh)
        errors2=np.zeros_like(x_mesh)
        errors3=np.zeros_like(x_mesh)
        #compute the errors
        for i in tqdm(range(x_mesh.shape[0]),desc=f"Computing errors"):
            for j in tqdm(range(x_mesh.shape[1]),desc=f"Computing errors at time {t_grid[i]}"):
                x_values=np.random.normal(0,1,(n_samples,self.dim))
                # print(np.linalg.norm(x_values,axis=1)[:,np.newaxis])
                # print(x_mesh[i,j])
                x_values/=np.linalg.norm(x_values,axis=1)[:,np.newaxis]
                x_values*=x_mesh[i,j]
                t_values = np.full((n_samples, 1), t_mesh[i, j]) # Create a 2D array filled with t_mesh[i, j]
                xt_values=np.concatenate((x_values,t_values),axis=1)
                exact_sol=eq.exact_solution(xt_values)
                '''A little bug: sol1 is float32 while sol2 and sol3 are float64'''
                sol1=self.solver1(torch.tensor(xt_values,dtype=torch.float32)).detach().numpy()[:,0]
                sol2=self.solver2.uz_solve(n,rho,xt_values)[:,0]
                sol3=self.solver3.u_solve(n,rho,xt_values)
                errors1[i,j]+=np.mean(sol1-exact_sol)
                errors2[i,j]+=np.mean(sol2-exact_sol)
                errors3[i,j]+=np.mean(sol3-exact_sol)
        # compute |errors1|-|errors3|,|errrors2|-|errors3|
        errors_13=np.abs(errors1)-np.abs(errors3)
        errors_23=np.abs(errors2)-np.abs(errors3)
        plt.figure()
        plt.imshow(errors1,extent=[0,self.radius,self.t0,self.T],aspect='auto',cmap='RdBu_r')
        plt.colorbar()
        plt.title("PINN error")
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        #display the nanmean of the errors
        plt.text(0.5,0.5,"nan sum={:.2f}".format(np.nanmean(errors1)),horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
        plt.savefig(f"{save_path}/PINN_error.png")
        plt.figure()
        plt.imshow(errors2,extent=[0,self.radius,self.t0,self.T],aspect='auto',cmap='RdBu_r')
        plt.colorbar()
        plt.title("MLP error, rho={:d}".format(rho))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        #display the nanmean of the errors
        plt.text(0.5,0.5,"nan sum={:.2f}".format(np.nanmean(errors2)),horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
        plt.savefig(f"{save_path}/MLP_error.png")
        plt.figure()
        plt.imshow(errors3,extent=[0,self.radius,self.t0,self.T],aspect='auto',cmap='RdBu_r')
        plt.colorbar()
        plt.title("ScaML error, rho={:d}".format(rho))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        #display the nanmean of the errors
        plt.text(0.5,0.5,"nan sum={:.2f}".format(np.nanmean(errors3)),horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
        plt.savefig(f"{save_path}/ScaML_error.png")
        plt.figure()
        plt.imshow(errors_13,extent=[0,self.radius,self.t0,self.T],aspect='auto',cmap='RdBu_r')
        plt.colorbar()
        plt.title("|PINN error| - |ScaML error|, rho={:d}".format(rho))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        #display the positive count and negative count of the difference of the errors
        plt.text(0.5,0.5,"positive count={:.2f}, negative count={:.2f}".format(np.sum(errors_13>0),np.sum(errors_13<0),horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes))   
        plt.savefig(f"{save_path}/PINN_ScaML_error.png")
        plt.figure()
        plt.imshow(errors_23,extent=[0,self.radius,self.t0,self.T],aspect='auto',cmap='RdBu_r')
        plt.colorbar()
        plt.title("|MLP error| - |ScaML error|, rho={:d}".format(rho))
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        #display the positive count and negative count of the difference of the errors
        plt.text(0.5,0.5,"positive count={:.2f}, negative count={:.2f}".format(np.sum(errors_23>0),np.sum(errors_23<0),horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes))
        plt.savefig(f"{save_path}/MLP_ScaML_error.png")
        wandb.log({"PINN error": wandb.Image(f"{save_path}/PINN_error.png"), "MLP error": wandb.Image(f"{save_path}/MLP_error.png"), "ScaML error": wandb.Image(f"{save_path}/ScaML_error.png")})
        wandb.log({"PINN-ScaML error": wandb.Image(f"{save_path}/PINN_ScaML_error.png"), "MLP-ScaML error": wandb.Image(f"{save_path}/MLP_ScaML_error.png")})

