import torch
import deepxde as dde

class FNN(object):
    '''FNN structure network based on deepxde framework'''
    def __init__(self, n_input,n_output):
        #initialize the FNN parameters
        self.n_input = n_input
        self.n_output = n_output
    def 