import torch
import numpy as np
import deepxde as dde
import torch.nn as nn
from scipy.special import lambertw

class ScaML(object):
    '''Multilevel Picard Iteration calibrated PINN for high dimensional semilinear PDE'''
    #all the vectors uses rows as index and columns as dimensions
    def __init__(self, equation,net):
        #initialize the ScaML parameters
        self.equation=equation
        self.sigma=equation.sigma
        self.mu=equation.mu
        self.T=equation.T
        self.t0=equation.t0
        self.n_input=equation.n_input
        self.n_output=equation.n_output
        net.eval()
        self.net=net


    def approx_PDE_loss(self,x_t,u_hat,grad_u_hat_x,dt):
        # PDE loss by plug in, using disretization instead of autodiff to accelerate the inferrence
        # must choose a defualt value for dt
        raise NotImplementedError
    
    def f(self,x_t,u_breve,z_breve):
        # generator of ScaML
        eq=self.equation
        tensor_x_t=torch.tensor(x_t,requires_grad=True).float()
        u_hat=self.net(tensor_x_t).detach().numpy()
        grad_u_hat_x=torch.autograd.grad(u_hat,tensor_x_t,grad_outputs=torch.ones_like(u_hat),create_graph=True)[0][:, :-1].detach().numpy()
        epsilon=self.approx_PDE_loss(x_t,u_hat,grad_u_hat_x)
        val1=eq.f(x_t,u_breve+u_hat,z_breve+eq.sigma(x_t)*grad_u_hat_x)  
        val2=eq.f(x_t,u_hat,eq.sigma(x_t)*grad_u_hat_x)
        return val1-val2-epsilon
    def g(self,x_t):
        # terminal constraint of ScaML
        eq=self.equation
        tensor_x_t=torch.tensor(x_t,requires_grad=True).float()
        u_hat=self.net(tensor_x_t).detach().numpy()
        result=eq.g(x_t)-u_hat
        return result
    
    def inverse_gamma(self, gamma_input):
        #inverse gamma function
        c = 0.036534 # avoid singularity
        L = np.log((gamma_input+c) / np.sqrt(2 * np.pi)) 
        return np.real(L / lambertw(L / np.e) + 0.5) # inverse gamma function
    
    def lgwt(self,N, a, b):
        #Legendre-Gauss nodes and weights
        N-=1 # truncation number
        N1, N2 = N+1, N+2 # number of nodes and weights
        xu = np.linspace(-1, 1, N1).reshape(1,-1) # uniform on [-1, 1], and transpose to row vector
        y = np.cos((2 * np.arange(0, N+1, 1)+ 1) * np.pi / (2 * N + 2))+(0.27/N1) * np.sin(np.pi * xu * N / N2) # initial guess
        L = np.zeros((N1, N2)) # Legendre-Gauss Vandermonde Matrix
        Lp = np.zeros((N1, N2)) # Derivative of Legendre-Gauss Vandermonde Matrix
        y0 = 2 
        # compute the zeros of the N+1 Legendre Polynomial
        # using the recursion relation and the Newton-Raphson method
        while np.max(np.abs(y-y0)) > 2.2204e-16: # iterate until new points are uniformly within epsilon of old points
            L[:, 0] = 1 
            Lp[:, 0] = 0
            L[:, 1] = y 
            Lp[:, 1] = 1
            for k in range(2, N1+1):
                L[:, k] = ((2 * k -1)* y * L[:,k-1]-(k-1)*L[:, k-2]) / k
            Lp = (N2) * (L[:, N1-1]-y * L[:, N2-1])/(1-y * y)
            y0 = y
            y = y0 - L[:, N2-1] / Lp
        x = (a * (1-y) + b * (1+y)) / 2 # linear map from [-1, 1] to [a, b]
        w = (b-a) / ((1-y*y) * Lp * Lp) * N2 * N2 / (N1 * N1) # compute weights
        return x[0], w[0]

    def approx_parameters(self, rhomax):
        # approximate parameters for the MLP
        levels = np.arange(1, rhomax+1)  # level list
        Q = np.zeros((rhomax, rhomax))  # number of quadrature points
        Mf = np.zeros((rhomax, rhomax))  # number of forward Euler steps
        Mg = np.zeros((rhomax, rhomax+1))  # number of backward Euler steps

        rho_values = np.arange(1, rhomax+1).reshape(-1, 1)
        k_values = np.arange(1, rhomax+1)

        Q = np.round(self.inverse_gamma(rho_values ** (k_values / 2)))
        Mf = np.round(rho_values ** (k_values / 2))
        Mg[:, :-1] = np.round(rho_values ** (k_values - 1))
        Mg[:, -1] = rho_values.flatten() ** rho_values.flatten()

        qmax = int(np.max(Q))  # maximum number of quadrature points
        c = np.zeros((qmax, qmax))  # quadrature points
        w = np.zeros((qmax, qmax))  # quadrature weights

        k_values = np.arange(1, qmax+1)
        ctemp, wtemp = self.lgwt(k_values, 0, self.T)  # Legendre-Gauss nodes and weights
        c = np.concatenate([ctemp[::-1], np.zeros((qmax, qmax - len(k_values)))], axis=0)  # quadrature points
        w = np.concatenate([wtemp[::-1], np.zeros((qmax, qmax - len(k_values)))], axis=0)  # quadrature weights

        return Mf, Mg, Q, c, w
        
    def u_breve_z_breve_solve(self,n, rho, x_t): 
        #approximate the solution of the PDE, return the value of u_breve(x_t) and z_breve(x_t), batchwisely
        #n: backward Euler samples needed
        #rho: current level
        #x_t: a batch of spatial-temporal coordinates
        Mf, Mg, Q, c, w = self.approxparameters(rho)
        T=self.T #terminal time
        dim=self.n_input-1 #spatial dimensions
        batch_size=x_t.shape[0] #batch size
        sigma=self.sigma(x_t) #volatility
        x=x_t[:, :-1] #spatial coordinates
        t=x_t[:, -1] #temporal coordinates
        f=self.f #generator term
        g=self.g #terminal constraint
        cloc = (T-t)[:, np.newaxis,np.newaxis] * c[np.newaxis,:]/T #local time
        wloc = (T-t)[:, np.newaxis,np.newaxis] * w[np.newaxis,:]/T #local weights
        MC = int(Mg[rho-1, n]) # number of monte carlo samples for backward Euler
        W = np.sqrt(T-t)[:, np.newaxis] * np.random.normal(size=(batch_size,MC, dim))
        X = np.repeat(x, MC, axis=1) + sigma * W
        u_breve = np.mean(np.apply_along_axis(g, 1, X), axis=1)
        z_breve = np.sum((np.repeat((np.apply_along_axis(g, 1, X)-g(x)[:, np.newaxis,:])[:,0], dim, axis=-1) * W), axis=1) / (MC * (T-t)[:, np.newaxis])
        if n <= 0:
            return np.concatenate((u_breve, z_breve),axis=-1)
        for l in range(n):
            q = int(Q[rho-1, n-l-1]) # number of quadrature points
            d = cloc[:,:q, q-1] - np.concatenate((t, cloc[:,:q-1, q-1]),axis=1) # time step
            MC = int(Mf[rho-1, n-l-1])
            X = np.repeat([x], MC, axis=1)
            W = np.zeros((batch_size,MC, dim))
            for k in range(q):
                dW = np.sqrt(d[:,k]) * np.random.normal(size=(batch_size, MC, dim))
                W += dW           
                X += sigma * dW            
                simulated=np.apply_along_axis(lambda x_t:self.u_breve_z_breve_solve(n=l,rho=rho,x_t=x_t),1,np.concatenate((X, np.repeat(cloc[:,k,q-1],MC,axis=1)),axis=-1))
                simulated_u_breve, simulated_z_breve = simulated[:,:, 0].reshape(batch_size,MC,1), simulated[:,:, 1:] 
                y = np.array( [f(cloc[:,k,q-1], X[:,i,:], simulated_u_breve[:,i,:], simulated_z_breve[:,i,:]) for i in range(MC)])
                y=y.transpose(1,0,2)
                u_breve += wloc[:,k, q-1] * np.mean(y, axis=1) 
                z_breve += wloc[:,k, q-1] * np.sum(np.repeat(y[:,:,0], dim, axis=-1) * W, axis=1) / (MC * (cloc[:,k, q-1]-t)[:,np.newaxis])
                if l:
                    simulated=np.apply_along_axis(lambda x_t:self.u_breve_z_breve_solve(n=l-1,rho=rho,x_t=x_t),1,np.concatenate((X, np.repeat(cloc[:,k,q-1],MC,axis=1)),axis=-1))
                    simulated_u_breve, simulated_z_breve = simulated[:,:, 0].reshape(batch_size,MC,1), simulated[:,:, 1:] 
                    y = np.array( [f(cloc[:,k,q-1], X[:,i,:], simulated_u_breve[:,i,:], simulated_z_breve[:,i,:]) for i in range(MC)])
                    y=y.transpose(1,0,2)
                    u_breve -= wloc[:,k, q-1] * np.mean(y, axis=1) 
                    z_breve -= wloc[:,k, q-1] * np.sum(np.repeat(y[:,:,0], dim, axis=-1) * W, axis=1) / (MC * (cloc[:,k, q-1]-t)[:,np.newaxis])
        return np.concatenate((u_breve, z_breve),axis=-1)
    def u_solve(self,n, rho, x_t): 
        #approximate the solution of the PDE, return the value of u(x_t) and z(x_t), batchwisely
        #n: backward Euler samples needed
        #rho: current level
        #x_t: a batch of spatial-temporal coordinates, ndarray
        u_breve_z=self.u_breve_z_breve_solve(n, rho, x_t)
        u_breve,z_breve=u_breve_z[:,0],u_breve_z[:,1:]
        eq=self.equation
        tensor_x_t=torch.tensor(x_t,requires_grad=True).float()
        u_hat=self.net(tensor_x_t).detach().numpy()
        u=u_breve+u_hat
        return u