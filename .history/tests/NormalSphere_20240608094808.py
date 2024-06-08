import numpy as np
import matplotlib.pyplot as plt
import wandb
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
        for i in range(x_mesh.shape[0]):
            for j in range(x_mesh.shape[1]):
                x_values=np.random.normal(0,1,(n_samples,self.dim))
                x_values/=np.linalg.norm(x_values,axis=1)[:,np.newaxis]*x_mesh[i,j]
                exact_sol=eq.exact_solution(np.concatenate((x_values,[t_mesh[i,j]]))[np.newaxis,:])
                sol1=self.solver1(np.concatenate((x_values,[t_mesh[i,j]]))[np.newaxis,:])
                sol2=self.solver2.uz_solve(n,rho,np.concatenate((x_values,[t_mesh[i,j]]))[np.newaxis,:])[:,0]
                sol3=self.solver3.u_solve(n,rho,np.concatenate((x_values,[t_mesh[i,j]]))[np.newaxis,:])
                errors1[i,j]+=np.mean(np.linalg.norm(sol1-exact_sol,axis=1))
                errors2[i,j]+=np.mean(np.linalg.norm(sol2-exact_sol,axis=1))
                errors3[i,j]+=np.mean(np.linalg.norm(sol3-exact_sol,axis=1))
        # compute 
        errors_12=np.abs(er)
        plt.figure()
        plt.imshow(errors1,extent=[0,self.radius,self.t0,self.T],aspect='auto',cmap='RdBu_r')
        plt.colorbar()
        plt.title("PINN error")
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/PINN_error.png")
        plt.figure()
        plt.imshow(errors2,extent=[0,self.radius,self.t0,self.T],aspect='auto',cmap='RdBu_r')
        plt.colorbar()
        plt.title("MLP error")
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/MLP_error.png")
        plt.figure()
        plt.imshow(errors3,extent=[0,self.radius,self.t0,self.T],aspect='auto',cmap='RdBu_r')
        plt.colorbar()
        plt.title("ScaML error")
        plt.xlabel("distance from origin")
        plt.ylabel("time")
        plt.savefig(f"{save_path}/ScaML_error.png")
        wandb.log({"PINN error": wandb.Image(f"{save_path}/PINN_error.png"), "MLP error": wandb.Image(f"{save_path}/MLP_error.png"), "ScaML error": wandb.Image(f"{save_path}/ScaML_error.png")})

