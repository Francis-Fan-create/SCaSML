import torch
import deepxde as dde

class FNN(object):
    '''FNN structure network bas'''
    def __init__(self, n_input, n_output,net,have_exact_solution=False):
        #initialize the equation parameters
        self.n_input = n_input