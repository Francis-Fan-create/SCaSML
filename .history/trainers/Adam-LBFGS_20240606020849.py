from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ExponentialLR

class Adam_LBFGS(object):
    '''Adam-LBFGS optimizer'''
    def __init__(self, net,data):
        #initialize the optimizer parameters
        self.net=net
        self.data=data
        self.model=dde.Model(data,net)

    def Adam(self, lr=1e-2):
        #Adam optimizer
        adam = Adam(self.net.parameters(), lr=lr)
        scheduler = ExponentialLR(adam, gamma=0.9)
        return adam
    def LBFGS(self, lr=1e-2lr=1e-2,max_iter=1000,tolerance_change=1e-5,tolerance_grad=1e-3):
        #LBFGS optimizer
        lbfgs = LBFGS(self.net.parameters(), lr=lr)
        return lbfgs