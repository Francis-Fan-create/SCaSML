



from equations.equations import Explict_Solution_Example
from models.FNN import FNN
from optimizers.Adam_LBFGS import Adam_LBFGS
from tests.NormalSphere import NormalSphere
from solvers.MLP import MLP
from solvers.ScaML import ScaML
import numpy as np
import torch
import wandb
import cProfile
import deepxde as dde

#fix random seed for dde
dde.config.set_random_seed(1234)
#use pytorch backend
dde.backend.set_default_backend('pytorch')
# fix random seed for numpy
np.random.seed(1234)
#set default data type
torch.set_default_dtype(torch.float64)
# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device.type == 'cuda':
    # get GPU name
    gpu_name = torch.cuda.get_device_name()

#initialize wandb
wandb.init(project="Explicit_Solution_Example")
wandb.config.update({"device": device.type}) # record device type

#initialize the equation
equation=Explict_Solution_Example(n_input=101,n_output=1)
data=equation.generate_data()
#initialize the FNN
layers=[101]+[50]*5+[1]
net=FNN(layers,equation)
#initialize the optimizer
optimizer=Adam_LBFGS(101,1,net,data)
#train the model
trained_model=optimizer.train("results/Explicit_Solution_Example/model.pth",cycle=40,adam_every=500,lbfgs_every=10,metrics=["l2 relative error","mse"])

#initialize the normal sphere test
solver1=trained_model.net #PINN network
solver1.eval()
solver2=MLP(equation=equation) #Multilevel Picard object
solver3=ScaML(equation=equation,net=solver1) #ScaML object
profiler = cProfile.Profile()
profiler.enable()
#run the test
test=NormalSphere(equation,solver1,solver2,solver3)
test.test("results/Explicit_Solution_Example")
#stop the profiler
profiler.disable()
#save the profiler results
profiler.dump_stats("results/Explicit_Solution_Example/explicit_solution_example.prof")
#upload the profiler results to wandb
artifact=wandb.Artifact("explicit_solution_example_profiler", type="profile")
artifact.add_file("results/Explicit_Solution_Example/explicit_solution_example.prof")
wandb.log_artifact(artifact)


#finish wandb
wandb.finish()
