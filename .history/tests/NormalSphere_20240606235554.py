import numpy as np
import matplotlib.pyplot as plt

class NormalShperes(object):
    '''Normal spheres in high dimensions'''
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
    def test(self,n_samples=10,x_grid_num=100,t_grid_num=10):
        #compare solvers on different distances on the sphere
        eq=self.equation
        x_grid=np.linspace(0,self.radius,x_grid_num)
        t_grid=np.linspace(self.t0,self.T,t_grid_num)
        x_mesh,t_mesh=np.meshgrid(x_grid,t_grid)
        errors=np.zeros_like(x_mesh)
        for i in range(x_mesh.shape[0]):
            for j in range(x_mesh.shape[1]):
                x_values=np.random.normal(0,1,(n_samples,self.dim))
                x_values/=np.linalg.norm(x_values,axis=1)[:,np.newaxis]*x_mesh[i,j]
                error_sum=0
                for x in x_values:
                    exact_sol=eq.exact_solution(np.concatenate((x,[t_mesh[i,j]]))[np.newaxis,:])
                    sol1=self.solver1(np.concatenate((x,[t_mesh[i,j]]))[np.newaxis,:])
                