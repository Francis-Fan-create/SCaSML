from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ExponentialLR
import deepxde as dde

class Adam_LBFGS(object):
    '''Adam-LBFGS optimizer'''
    def __init__(self,n_input,n_output,net,data):
        #initialize the optimizer parameters
        self.net=net
        self.data=data
        self.n_input=n_input
        self.n_output=n_output
        self.model=dde.Model(data,net)

    def Adam(self, lr=1e-2,weight_decay=1e-4,gamma=0.9):
        #Adam optimizer
        adam = Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ExponentialLR(adam, gamma=gamma)
        return adam
    def LBFGS(self, lr=1e-2,max_iter=1000,tolerance_change=1e-5,tolerance_grad=1e-3):
        #LBFGS optimizer
        lbfgs = LBFGS(self.net.parameters(), lr=lr, max_iter=max_iter, tolerance_change=tolerance_change, tolerance_grad=tolerance_grad)
        return lbfgs
    
    def train(self,loss_weights,cycle=20,adam_every=1000,lbfgs_every=10,metrics=["l2 relative error","mse"]):
        #interleaved training of adam and lbfgs
        for i in range(cycle):
            self.model.compile(optimizer=self.Adam(),metrics=metrics,loss_weights=loss_weights)
            self.model.train(epochs=adam_every, display_every=10, metrics=metrics)
            self.model.compile()
            self.model.train(epochs=lbfgs_every, display_every=10, metrics=metrics)
        return self.model