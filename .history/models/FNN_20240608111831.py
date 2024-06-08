import torch
import deepxde as dde
import torch.nn as nn
import wandb
class FNN(dde.maps.pytorch.M):
    '''FNN structure network based on deepxde framework'''
    def __init__(self, layers,equation):
        #initialize the FNN parameters
        super(FNN, self).__init__()
        self.layers = layers #layer list of the FNN
        self.net=dde.maps.pytorch.FNN(layers, "tanh", "Glorot normal") #initialize the FNN
        self.equation=equation
        self.regularizer=None
        #we do not need to initialize wandb here, as it is already initialized in the main script
        wandb.config.update({"layers": layers}) # record hyperparameters
        wandb.watch(self.net) #watch the FNN on wandb
    def forward(self,x_t):
        #forward pass of the FNN
        #x_t is the input tensor
        eq=self.equation
        before_transform=self.net(x_t)  #forward pass
        after_transform=eq.terminal_condition(x_t,before_transform) #apply terminal condition
        return after_transform