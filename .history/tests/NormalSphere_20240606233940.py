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
    def test(self,n_samples=10,x_grid_num=100):
        #compare solvers on different distances on the sphere
        x_grid=np.linspace(self.t0,self.T,100).reshape(-1,1)