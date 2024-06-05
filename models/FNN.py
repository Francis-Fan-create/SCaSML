import torch
import deepxde as dde
import torch.nn as nn

class FNN(nn.Module):
    '''FNN structure network based on deepxde framework'''
    def __init__(self, layers):
        #initialize the FNN parameters
        super(FNN, self).__init__()
        self.layers = layers #layer list of the FNN
        self.net=dde.maps.pytorch.FNN(layers)
        self.regulerizer=None
    def forward(self,x_t):
        #forward pass of the FNN
        return self.net(x_t)