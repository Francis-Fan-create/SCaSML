from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ExponentialLR

class Adam_LBFGS:
    '''Adam-LBFGS optimizer'''
    def __init__(self, model, lr=1e-3, max_iter=4, max_eval=5, tolerance_grad=1e-5, tolerance_change=1e-9):
        #initialize the optimizer parameters
        self.model = model