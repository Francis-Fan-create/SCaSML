from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ExponentialLR

class Adam_LBFGS(object):
    '''Adam-LBFGS optimizer'''
    def __init__(self, net,data):
        #initialize the optimizer parameters
        self.model = model

    def Adam(self, lr):
        #Adam optimizer
        optimizer = Adam(self.model.parameters(), lr=lr)
        return optimizer