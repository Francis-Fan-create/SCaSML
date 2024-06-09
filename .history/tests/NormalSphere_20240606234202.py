import numpy as np
import matplotlib.pyplot as plt

class NormalShperes(object):
    '''Normal spheres in high dimensions'''
    def __init__(self, equation, solver1,solver2,solver3):
        #initialize the normal spheres
        #solver1 for PINN
        #solver2 for MLP
        #solver3 for ScaML
        self.equation=equation
        self.dim=equation.n_input-1
        self.solver1=solver1
        self.solver2=solver2
        self.solver3=solver3
        self.t0=equation.t0
        self.T=equation.T
        self.radius=np.sqrt(self.dim*(self.T-self.t0)**2)
    def test(self,n_samples=10,x_grid_num=100,t_grid_num=10):
        #compare solvers on different distances on the sphere
        x_grid=np.linspace(0,self.radius,x_grid_num)
        t_grid=np.linspace(self.t0,self.T,t_grid_num)
        x_mesh,t_mesh=np.meshgrid(x_grid,t_grid)
        errors=np.zeros_like(x_mesh)
        for i in range(x_mesh.shape[0]):
            for j in range(x_mesh.shape[1]):
                x_t=np.zeros((n_samples,self.dim+1))
                x_t[:,0:self.dim]=x_mesh[i,j]
                x_t[:,-1]=t_mesh[i,j]
                u_exact=self.equation.exact_solution(x_t)
                u1=self.solver1.u_solve(0,1,x_t)
                u2=self.solver2.u_solve(0,1,x_t)
                u3=self.solver3.u_solve(0,1,x_t)
                errors[i,j]=np.mean(np.abs(u1-u_exact)+np.abs(u2-u_exact)+np.abs(u3-u_exact))