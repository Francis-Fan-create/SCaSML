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
    
    def inverse_gamma(self, gamma_input):
        #inverse gamma function
        c = 0.036534
        L = np.log((gamma_input+c) / np.sqrt(2 * np.pi))
        return np.real(L / lambertw(L / np.e) + 0.5)
    
    def lgwt(self,N, a, b):
        #Legendre-Gauss nodes and weights
        N = N - 1 # number of nodes
        N1, N2 = N+1, N+2 # number of nodes and weights
        xu = np.linspace(-1, 1, N1).reshape(1,-1) # nodes
        y = np.cos((2 * np.arange(0, N+1, 1)+ 1) * np.pi / (2 * N + 2))+(0.27/N1) * np.sin(np.pi * xu * N / N2) # initial guess
        L = np.zeros((N1, N2)) # Legendre polynomial
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
        n = list(range(1, rhomax+1)) 
        Q = np.zeros((rhomax, rhomax)) #number of quadrature points
        Mf = np.zeros((rhomax, rhomax)) #number of forward Euler steps
        Mg = np.zeros((rhomax, rhomax+1)) #number of backward Euler steps
        for rho in range(1, rhomax+1):
            for k in range(1, n[rho-1]+1):
                Q[rho-1][k-1] = round(self.inverse_gamma(rho ** (k/2))) #inverse gamma function
                Mf[rho-1][k-1] = round(rho ** (k/2)) #forward Euler steps
                Mg[rho-1][k-1] = round(rho ** (k-1)) #backward Euler steps
            Mg[rho-1][rho] = rho ** rho #backward Euler steps
        qmax = int(np.max(Q)) #maximum number of quadrature points
        c = np.zeros((qmax, qmax)) #quadrature points
        w = np.zeros((qmax, qmax)) #quadrature weights
        for k in range(1, qmax+1):
            ctemp, wtemp = self.lgwt(k, 0, self.T) #Legendre-Gauss nodes and weights
            c[:, k-1] = np.concatenate([ctemp[::-1], np.zeros(qmax-k)]) #quadrature points
            w[:, k-1] = np.concatenate([wtemp[::-1], np.zeros(qmax-k)]) #quadrature weights
        self.Mf, self.Mg, self.Q, self.c, self.w, self.n = Mf, Mg, Q, c, w, n #save the parameters
        return Mf, Mg, Q, c, w, n
    
    def approximateUZpde(self,n, rho, x_t): 
        #approximate the solution of the PDE, return the value of u(x_t) and z(x_t)
        x=x_t[:, :-1]
        t=x_t[:, -1]
        cloc = (self.T-t).unsqueeze(-1) * self.c/self.T
        wloc = (self.T-t).unsqueeze(-1) * self.w/self.T
        MC = int(self.Mg[rho-1, n])
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

