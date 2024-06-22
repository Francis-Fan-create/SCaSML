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
        equation.geometry()
        self.T=equation.T
        self.t0=equation.t0
        self.n_input=equation.n_input
        self.n_output=equation.n_output
        net.eval()
        self.net=net
        '''A potential way to accelerate the inference process: use discretized version of laplacian'''
     
    def f(self,x_t,u_breve,z_breve):
        # generator of ScaML, this is the light and large version
        eq=self.equation
        tensor_x_t=torch.tensor(x_t,requires_grad=True).float()
        tensor_u_hat=self.net(tensor_x_t)
        u_hat=tensor_u_hat.detach().cpu().numpy()
        tensor_grad_u_hat_x=torch.autograd.grad(tensor_u_hat,tensor_x_t,grad_outputs=torch.ones_like(tensor_u_hat),retain_graph=True,create_graph=True)[0][:, :-1]
        grad_u_hat_x=tensor_grad_u_hat_x.detach().cpu().numpy()
        # epsilon=eq.PDE_loss(tensor_x_t,tensor_u_hat,tensor_grad_u_hat_x).detach().cpu().numpy()
        val1=eq.f(x_t,u_breve+u_hat,eq.sigma(x_t)*(grad_u_hat_x+z_breve))  
        val2=eq.f(x_t,u_hat,eq.sigma(x_t)*grad_u_hat_x)
        # return val1-val2-epsilon #large version
        return val1-val2 #light version
    def g(self,x_t):
        # terminal constraint of ScaML
        eq=self.equation
        tensor_x_t=torch.tensor(x_t,requires_grad=True).float()
        u_hat=self.net(tensor_x_t).detach().cpu().numpy()
        result=eq.g(x_t)-u_hat[:,0]
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

    def approx_parameters(self,rhomax):
        #approximate parameters for the MLP
        levels = list(range(1, rhomax+1)) #level list
        Q = np.zeros((rhomax, rhomax)) #number of quadrature points
        Mf = np.zeros((rhomax, rhomax)) #number of forward Euler steps
        Mg = np.zeros((rhomax, rhomax+1)) #number of backward Euler steps
        for rho in range(1, rhomax+1):
            for k in range(1, levels[rho-1]+1):
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
        return Mf, Mg, Q, c, w
    def set_approx_parameters(self,rhomax):
        #set the approximation parameters
        self.Mf, self.Mg, self.Q, self.c, self.w = self.approx_parameters(rhomax)

    def uz_solve(self,n, rho, x_t): 
        #approximate the solution of the PDE, return the value of u(x_t) and z(x_t), batchwisely
        #n: backward Euler samples needed
        #rho: current level
        #x_t: a batch of spatial-temporal coordinates, ndarray
        Mf, Mg, Q, c, w = self.Mf, self.Mg, self.Q, self.c, self.w
        T=self.T #terminal time
        dim=self.n_input-1 #spatial dimensions
        batch_size=x_t.shape[0] #batch size
        sigma=self.sigma(x_t) #volatility
        x=x_t[:, :-1] #spatial coordinates
        t=x_t[:, -1] #temporal coordinates
        f=self.f #generator term
        g=self.g #terminal constraint
        cloc = (T-t)[:, np.newaxis,np.newaxis] * c[np.newaxis,:]/T+t[:,np.newaxis,np.newaxis] #local time
        wloc = (T-t)[:, np.newaxis,np.newaxis] * w[np.newaxis,:]/T #local weights
        MC = int(Mg[rho-1, n]) # number of monte carlo samples for backward Euler
        W = np.sqrt(T-t)[:, np.newaxis,np.newaxis] * np.random.normal(size=(batch_size,MC, dim))
        X = np.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC, axis=1)
        disturbed_X = np.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC, axis=1) + sigma * W
        # print(X.shape)
        # print(X)
        terminals=np.zeros((batch_size,MC,1))
        differences=np.zeros((batch_size,MC,1))
        for i in range(MC):
            input_terminal = np.concatenate((X[:, i, :], np.full((batch_size, 1), T)), axis=1)
            disturbed_input_terminal= np.concatenate((disturbed_X[:, i, :], np.full((batch_size, 1), T)), axis=1)
            terminals[:,i,:]=g(input_terminal)[:,np.newaxis]
            differences[:,i,:]=(g(disturbed_input_terminal)-g(input_terminal))[:,np.newaxis]
        u = np.mean(differences+terminals,axis=1) 
        if (T-t).any()==0:
            delta_t=(T-t+1e-15)[:,np.newaxis]
            z = np.sum(differences*W,axis=1)/ (MC * delta_t)
        else:
            z = np.sum(differences*W,axis=1)/ (MC * (T-t)[:, np.newaxis])
        if n <= 0:
            return np.concatenate((u, z),axis=-1)
        for l in range(n):
            q = int(Q[rho-1, n-l-1]) # number of quadrature points
            d = cloc[:,:q, q-1] - np.concatenate((t[:,np.newaxis], cloc[:,:q-1, q-1]),axis=1) # time step
            MC = int(Mf[rho-1, n-l-1])
            X = np.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC, axis=1)
            W = np.zeros((batch_size,MC, dim))
            simulated=np.zeros((batch_size,MC,dim+1))
            for k in range(q):
                dW = np.sqrt(d[:,k])[:,np.newaxis,np.newaxis] * np.random.normal(size=(batch_size, MC, dim))
                W += dW           
                X += sigma * dW 
                co_solver_l=lambda X_t:self.uz_solve(n=l,rho=rho,x_t=X_t)
                co_solver_l_minus_1=lambda X_t:self.uz_solve(n=l-1,rho=rho,x_t=X_t)
                input_intermediates=np.zeros((batch_size,MC,dim+1))
                for i in range(MC):
                    input_intermediate= np.concatenate((X[:, i, :], cloc[:,k,q-1][:,np.newaxis]), axis=1)
                    simulated[:,i,:]=co_solver_l(input_intermediate)          
                    input_intermediates[:,i,:]=input_intermediate
                simulated_u, simulated_z = simulated[:,:, 0].reshape(batch_size,MC,1), simulated[:,:, 1:] 
                y = np.array( [f(input_intermediates[:,i,:], simulated_u[:,i,:], simulated_z[:,i,:]) for i in range(MC)])
                y=y.transpose(1,0,2)
                u += wloc[:,k, q-1][:,np.newaxis] * np.mean(y, axis=1)
                if (cloc[:,k, q-1]-t).any()==0:
                    delta_t=(cloc[:,k, q-1]-t+1e-15)[:,np.newaxis]
                    z += wloc[:,k, q-1][:,np.newaxis] * np.sum(y*W, axis=1) / (MC * delta_t)
                else:
                    z += wloc[:,k, q-1][:,np.newaxis] * np.sum(y*W, axis=1) / (MC * (cloc[:,k, q-1]-t)[:,np.newaxis])
                if l:
                    input_intermediates=np.zeros((batch_size,MC,dim+1))
                    for i in range(MC):
                        input_intermediate= np.concatenate((X[:, i, :], cloc[:,k,q-1][:,np.newaxis]), axis=1)
                        simulated[:,i,:]=co_solver_l_minus_1(input_intermediate)          
                        input_intermediates[:,i,:]=input_intermediate
                    simulated_u, simulated_z = simulated[:,:, 0].reshape(batch_size,MC,1), simulated[:,:, 1:] 
                    y = np.array( [f(input_intermediates[:,i,:], simulated_u[:,i,:], simulated_z[:,i,:]) for i in range(MC)])
                    y=y.transpose(1,0,2)
                    u -= wloc[:,k, q-1][:,np.newaxis] * np.mean(y, axis=1)
                    if (cloc[:,k, q-1]-t).any()==0:
                        delta_t=(cloc[:,k, q-1]-t+1e-15)[:,np.newaxis]
                        z -= wloc[:,k, q-1][:,np.newaxis] * np.sum(y*W, axis=1) / (MC * delta_t)
                    else:
                        z -= wloc[:,k, q-1][:,np.newaxis] * np.sum(y*W, axis=1) / (MC * (cloc[:,k, q-1]-t)[:,np.newaxis])
        return np.concatenate((u, z),axis=-1)
    
    def u_solve(self,n, rho, x_t): 
        #approximate the solution of the PDE, return the value of u(x_t) and z(x_t), batchwisely
        #n: backward Euler samples needed
        #rho: current level
        #x_t: a batch of spatial-temporal coordinates, ndarray
        u_breve_z_breve=self.uz_solve(n, rho, x_t)
        u_breve,z_breve=u_breve_z_breve[:,0],u_breve_z_breve[:,1:]
        tensor_x_t=torch.tensor(x_t,requires_grad=True).float()
        u_hat=self.net(tensor_x_t).detach().cpu().numpy()[:,0]
        u=u_breve+u_hat
        return u
    
