from equations.equations import Explict_Solution_Example
from models.FNN import FNN
from optimizers.Adam_LBFGS import Adam_LBFGS
from tests.NormalSphere import NormalSphere
import numpy as np
import torch
import wandb
import cProfile
import deepxde as dde


dde.config.set_random_seed(1234)
dde.backend.set_default_backend('pytorch')
# fix random seed for numpy
np.random.seed(1234)
#set default data type
torch.set_default_dtype(torch.float64)
# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name())
#initialize wandb
wandb.init(project="Explicit_Solution_Example")
