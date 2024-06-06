import torch
import numpy as np
import deepxde as dde
import torch.nn as nn
from scipy.special import lambertw

class MLP(object):
    '''Multilevel Picard Iteration for high dimensional semilinear PDE'''
    def __init__(self, equation):
        #initialize the MLP parameters
        self.equation=equation
        self.sigma=equation.sigma
        self.mu=equation.mu
        self.T=equation.T
        self.t0=equation.t0
        self.n_input=equation.n_input
        self.n_output=equation.n_output
    
    def inverse_gamma(self, x):
        #inverse gamma function
        c=
    


