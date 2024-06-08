import torch
import deepxde as dde

class FNN(object):
    '''FNN structure network based on deepxde framework'''
    def __init__(self, n_input,n ,n_output):
        #initialize the equation parameters
        self.n_input = n_input