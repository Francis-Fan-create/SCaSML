from equations.equations import Explict_Solution_Example
from models.FNN import FNN
from optimizers.Adam_LBFGS import Adam_LBFGS
from tests.NormalSphere import NormalSphere
import numpy as np
import torch
import wandb
import cProfile

#initialize wandb
wandb.init
