import torch
import deepxde as dde
import torch.nn as nn
import wandb
class FNN(nn.Module):
    '''FNN structure network based on deepxde framework'''
    def __init__(self, layers,equation):
        #initialize the FNN parameters
        super(FNN, self).__init__()
        self.layers = layers #layer list of the FNN
        self.net=dde.maps.pytorch.FNN(layers, "tanh", "Gloro") #initialize the FNN
        self.equation=equation
        self.regulerizer=None
        #we do not need to initialize wandb here, as it is already initialized in the main script
        wandb.config.update({"layers": layers}) # record hyperparameters
        wandb.watch(self.net) #watch the FNN on wandb
    def forward(self,x_t):
        #forward pass of the FNN
        eq=self.equation
        result=eq.terminal_condition(x_t,self.net(x_t))
        return result