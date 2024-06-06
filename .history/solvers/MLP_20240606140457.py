import torch
import numpy as np
import deepxde as dde
import torch.nn as nn
from scipy.special import lambertw

class MLP(object):
    '''Multilevel Picard Iteration for high dimensional semilinear PDE'''
    def __init__(self, n_input, n_output, n_hidden, n_levels, activation, have_exact_solution):
        #initialize the MLP parameters
        self.n_input = n_input

