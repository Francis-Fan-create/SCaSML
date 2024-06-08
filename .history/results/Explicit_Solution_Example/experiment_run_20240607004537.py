from equations.equations import Explict_Solution_Example
from models.FNN import FNN
from optimizers.Adam_LBFGS import Adam_LBFGS
from tests.NormalSphere import NormalSphere
import numpy as np
import torch
import wandb
import cProfile

#initialize wandb
wandb.init(project="Explicit_Solution_Example", entity="authors",config={"Adam lr": 1e-2, "Adam weight_decay": 1e-4, "Adam gamma": 0.9, "LBFGS lr": 1e-2, "LBFGS max_iter": 1000, "LBFGS tolerance_change": 1e-5, "LBFGS tolerance_grad": 1e-3, "loss_weights": [1,1,1], "cycle": 40, "adam_every": 500, "lbfgs_every": 10, "layers": [2, 50, 50, 50, 50, 50, 50, 50, 50, 50, 1]})
