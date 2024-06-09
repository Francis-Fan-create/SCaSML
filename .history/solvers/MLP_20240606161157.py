import torch
import numpy as np
import deepxde as dde
import torch.nn as nn
from scipy.special import lambertw

class MLP(object):
    '''Multilevel Picard Iteration for high dimensional semilinear PDE'''
    #all the vectors uses rows as index and columns as dimensions
    def __init__(self, equation):
        #initialize the MLP parameters
        self.equation=equation
        self.sigma=equation.sigma
        self.mu=equation.mu
        self.f=equation.f
        self.g=equation.g
        self.T=equation.T
        self.t0=equation.t0
        self.n_input=equation.n_input
        self.n_output=equation.n_output
    
    def inverse_gamma(self, x):
        #inverse gamma function
        c = 0.036534
        L = np.log((x+c) / np.sqrt(2 * np.pi))
        return np.real(L / lambertw(L / np.e) + 0.5)
    
    def lgwt(self,N, a, b):
        #Legendre-Gauss nodes and weights
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

    def approxparameters(self,rhomax):
        #approximate parameters for the MLP
        n = [i for i in range(1, rhomax+1)]
        Q = np.zeros((rhomax, rhomax))
        Mf = np.zeros((rhomax, rhomax))
        Mg = np.zeros((rhomax, rhomax+1))
        for rho in range(1, rhomax+1):
            for k in range(1, n[rho-1]+1):
                Q[rho-1][k-1] = round(self.inverse_gamma(rho ** (k/2)))
                Mf[rho-1][k-1] = round(rho ** (k/2))
                Mg[rho-1][k-1] = round(rho ** (k-1))
            Mg[rho-1][rho] = rho ** rho
        qmax = int(np.max(Q))
        c = np.zeros((qmax, qmax))
        w = np.zeros((qmax, qmax))
        for k in range(1, qmax+1):
            ctemp, wtemp = self.lgwt(k, 0, T)
            c[:, k-1] = np.concatenate([ctemp[::-1], np.zeros(qmax-k)])
            w[:, k-1] = np.concatenate([wtemp[::-1], np.zeros(qmax-k)])

        return Mf, Mg, Q, c, w, n
    
    def approximateUZpde(n, rho, x, s): 
        #approximate the solution of the PDE, return the value of u and z
        cloc = (T-s) * c/T + s
        wloc = (T-s) * w/T
        MC = int(Mg[rho-1, n])
        W = np.sqrt(T-s) * np.random.normal(size=(MC, dim))
        X = np.repeat([x], MC, axis=0) + sigma * W
        u = np.array([np.sum(g_vec(X)) / MC])
        z = np.nansum(np.repeat(g_vec(X)-g(x) * np.ones((MC, 1)), dim, axis=1) * W, axis=0) / (MC * (T-s))
        if n <= 0:
            return np.concatenate((u, z))
        for l in range(n):
            q = int(Q[rho-1, n-l-1])
            d = cloc[:q, q-1] - np.concatenate((np.array([s]), cloc[:q-1, q-1]))
            MC = int(Mf[rho-1, n-l-1])
            X = np.repeat([x], MC, axis=0)
            W = np.zeros((MC, dim))
            for k in range(q):
                dW = np.sqrt(d[k]) * np.random.normal(size=(MC, dim))
                W += dW           
                X += sigma * dW            
                mat = np.array([approximateUZpde(l, rho, X[i], cloc[k, q-1]) for i in range(MC)])
                # mat = np.frompyfunc(func, 4, 1)
                u_matrix, z_matrix = mat[:, 0].reshape(-1,1), mat[:, 1:]# .reshape(-1,dim)
                y = f(cloc[k, q-1], X, u_matrix, z_matrix)
    #             print(y.shape)
                u += wloc[k, q-1] * np.sum(y, axis=0) / MC
                z += wloc[k, q-1] * np.sum(np.repeat(y, dim, axis=1) * W, axis=0) / (MC * (cloc[k, q-1]-s))
                if l:
                    mat = np.array([approximateUZpde(l-1, rho, X[i], cloc[k, q-1]) for i in range(MC)])
                    u_matrix, z_matrix = mat[:, 0].reshape(-1,1), mat[:, 1:]# .reshape(-1,dim)
                    y = f(cloc[k, q-1], X, u_matrix, z_matrix)
                    u -= wloc[k, q-1] * np.sum(y, axis=0) / MC
                    z -= wloc[k, q-1] * np.sum(np.repeat(y, dim, axis=1) * W, axis=0) / (MC * (cloc[k, q-1]-s))
        return np.concatenate((u, z))

