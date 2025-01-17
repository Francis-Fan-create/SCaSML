import torch
import deepxde as dde
import torch.nn as nn
import wandb

class FNN(nn.Module):
    '''FNN structure network based on deepxde framework.
    
    This class defines a feedforward neural network (FNN) structure using the deepxde framework and PyTorch. It is designed to work with differential equations by integrating the network with the deepxde library's functionalities.
    
    Attributes:
        layers (list): A list of integers specifying the number of neurons in each layer of the FNN.
        equation (Equation): An instance of an Equation class that defines the differential equation to be solved.
    '''
    
    def __init__(self, layers, equation):
        '''Initializes the FNN with specified layers and equation.
        
        Args:
            layers (list): A list of integers specifying the number of neurons in each layer of the FNN.
            equation (Equation): An instance of an Equation class that defines the differential equation to be solved.
        '''
        super(FNN, self).__init__()
        self.layers = layers  # layer list of the FNN
        self.net = dde.maps.pytorch.FNN(layers, "tanh", "Glorot normal")  # initialize the FNN for model 1
        # self.net = dde.maps.FNN(layers, "swish", "Glorot normal")  # initialize the FNN for model 2
        self.equation = equation
        self.regularizer = None
        # we do not need to initialize wandb here, as it is already initialized in the main script
        wandb.config.update({"layers": layers})  # record hyperparameters
        # wandb.watch(self.net, log="all", log_freq=60)  # watch the FNN model

    def forward(self, x_t):
        '''Performs a forward pass through the FNN.
        
        Args:
            x_t (Tensor): The input tensor with dimensions [batch_size, input_size].
        
        Returns:
            Tensor: The output tensor of the FNN with dimensions [batch_size, output_size].
        '''
        return self.net(x_t)