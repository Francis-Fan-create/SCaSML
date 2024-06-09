from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ExponentialLR

class Adam_LBFGS(object):
    '''Adam-LBFGS optimizer'''
    def __init__(self, model):
        #initialize the optimizer parameters
        self.model = model

    