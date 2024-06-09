import deepxde as dde
import numpy as np
import torch
from deepxde.backend import tensorflow as tf
from torch import nn
from torch.nn import functional as F


# set random seed of deepxde
dde.config.set_random_seed(1234)
# fix random seed for numpy
np.random.seed(1234)
# Set default dtype in PyTorch to float64
torch.set_default_dtype(torch.float64)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name())