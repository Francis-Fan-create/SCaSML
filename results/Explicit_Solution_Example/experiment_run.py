# We will use this file to run the experiment for the Explicit_Solution_Example equation. We will train the model and then test it on the NormalSphere test. We will also profile the test to see the time taken by each solver to solve the equation. We will save the profiler results and upload them to wandb.
import sys
import os
#add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


# import the required libraries
from equations.equations import Explict_Solution_Example
from models.FNN import FNN
from optimizers.Adam_LBFGS import Adam_LBFGS
from tests.NormalSphere import NormalSphere
from solvers.MLP import MLP
from solvers.ScaML import ScaML
import numpy as np
import torch
import wandb
import deepxde as dde
import cProfile
import io
import pstats

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
wandb.init(project="Explicit_Solution_Example", notes="100 d", tags=["normal sphere test","Adam-LBFGS training"],mode="disabled") #debug mode
# wandb.init(project="Explicit_Solution_Example", notes="100 d", tags=["normal sphere test","Adam-LBFGS training"]) #working mode
wandb.config.update({"device": device.type}) # record device type

#initialize the equation
equation=Explict_Solution_Example(n_input=101,n_output=1)
#check if trained model is already saved
if os.path.exists(r"results/Explicit_Solution_Example/model_weights_1.params"):
    #load the model
    net=FNN([101]+[50]*5+[1],equation)
    net.load_state_dict(torch.load(r"results/Explicit_Solution_Example/model_weights_1.params",map_location=device)) #the other indexes are left for external resources of weights
    trained_net=net
else:
    data=equation.generate_data()
    #initialize the FNN
    layers=[101]+[50]*5+[1]
    net=FNN(layers,equation)
    #initialize the optimizer
    optimizer=Adam_LBFGS(101,1,net,data)
    #train the model
    # trained_model=optimizer.train("results/Explicit_Solution_Example/model_weights_1.params")
    trained_model=optimizer.train("results/Explicit_Solution_Example/model_weights_2.params")   
    trained_net=trained_model.net


#initialize the normal sphere test
solver1=trained_net #PINN network
solver1.eval()
solver2=MLP(equation=equation) #Multilevel Picard object
solver3=ScaML(equation=equation,net=solver1) #ScaML object

#initialize the profiler
profiler = cProfile.Profile()
profiler.enable()
#run the test
test=NormalSphere(equation,solver1,solver2,solver3)
rhomax=test.test(r"results/Explicit_Solution_Example")
#stop the profiler
profiler.disable()
#save the profiler results
profiler.dump_stats(f"results/Explicit_Solution_Example/explicit_solution_example_profiler_rho_{rhomax}.prof")
#upload the profiler results to wandb
artifact=wandb.Artifact(f"explicit_solution_example_profiler_rho_{rhomax}.prof", type="profile")
artifact.add_file(f"results/Explicit_Solution_Example/explicit_solution_example_profiler_rho_{rhomax}.prof")
wandb.log_artifact(artifact)

# Create a StringIO object to redirect the profiler output
s = io.StringIO()
# Create a pstats.Stats object and print the stats
stats = pstats.Stats(profiler, stream=s)
stats.print_stats()
# Print the profiler output
print(s.getvalue())



#finish wandb
wandb.finish()
