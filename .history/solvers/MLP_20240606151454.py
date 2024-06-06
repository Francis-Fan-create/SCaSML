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
        c = 0.036534
        L = np.log((x+c) / np.sqrt(2 * np.pi))
        return np.real(L / lambertw(L / np.e) + 0.5)
    
    def lgwt(N, a, b):
    N = N - 1
    N1, N2 = N+1, N+2
    xu = np.linspace(-1, 1, N1).reshape(1,-1)
    y = np.cos((2 * np.arange(0, N+1, 1)+ 1) * np.pi / (2 * N + 2))+(0.27/N1) * np.sin(np.pi * xu * N / N2)
    L = np.zeros((N1, N2))
    Lp = np.zeros((N1, N2))
    y0 = 2
    while np.max(np.abs(y-y0)) > 2.2204e-16:
        L[:, 0] = 1
        Lp[:, 0] = 0
        L[:, 1] = y
        Lp[:, 1] = 1
        for k in range(2, N1+1):
            L[:, k] = ((2 * k -1)* y * L[:,k-1]-(k-1)*L[:, k-2]) / k
        Lp = (N2) * (L[:, N1-1]-y * L[:, N2-1])/(1-y * y)
        y0 = y
        y = y0 - L[:, N2-1] / Lp
    x = (a * (1-y) + b * (1+y)) / 2
    w = (b-a) / ((1-y*y) * Lp * Lp) * N2 * N2 / (N1 * N1)
    return x[0], w[0]


