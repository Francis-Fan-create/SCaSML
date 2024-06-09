import torch
import numpy as np
import deepxde as dde
import torch.nn as nn
from scipy.special import lambertw

class MLP(object):
    '''Multilevel Picard Iteration for high dimensional semilinear PDE'''
    def __init__(self, equation,T):
        #initialize the MLP parameters
        self.equation=equation
        self.sigma=equation.sigma
        self.mu=equation.mu


