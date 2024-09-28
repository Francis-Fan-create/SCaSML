# We will use this file to run the experiment. We will train the model and then test it on the NormalSphere test. We will also profile the test to see the time taken by each solver to solve the equation. We will save the profiler results_full_history and upload them to wandb.
import sys
import os
#add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# import the required libraries
from equations.equations import Neumann_Boundary
from models.FNN import FNN
from optimizers.Adam_LBFGS import Adam_LBFGS
from tests.NormalSphere import NormalSphere
from tests.SimpleUniform import SimpleUniform
from tests.ConvergenceRate import ConvergenceRate
from solvers.MLP_full_history  import MLP_full_history as MLP
from solvers.ScaSML_full_history  import ScaSML_full_history as ScaSML
import numpy as np
import torch
import wandb
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
wandb.init(project="Neumann_Boundary", notes="100 d", tags=["Adam_LBFGS training"],mode="disabled") #debug mode
# wandb.init(project="Neumann_Boundary", notes="100 d", tags=["Adam_LBFGS training"]) #working mode
wandb.config.update({"device": device.type}) # record device type

#initialize the equation
equation=Neumann_Boundary(n_input=101,n_output=1)
#check if trained model is already saved
if os.path.exists(r"results_full_history/Neumann_Boundary/100d/model_weights_Adam_LBFGS.params"):
    '''To Do: Retrain the model with new data points& Try new methods to reduce errors'''
    #load the model
    net=FNN([101]+[50]*5+[1],equation)
    net.load_state_dict(torch.load(r"results_full_history/Neumann_Boundary/100d/model_weights_Adam_LBFGS.params",map_location=device)) #the other indexes are left for external resources of weights
    trained_net=net
else:
    data=equation.generate_data()
    #initialize the FNN
    layers=[101]+[50]*5+[1]
    net=FNN(layers,equation)
    #initialize the optimizer
    optimizer=Adam_LBFGS(101,1,net,data,equation) #Adam-LBFGS optimizer
    #train the model
    trained_model=optimizer.train(r"results_full_history/Neumann_Boundary/100d/model_weights_Adam_LBFGS.params")
    trained_net=trained_model.net


#initialize the normal sphere test
solver1=trained_net #PINN network
solver1.eval()
solver2=MLP(equation=equation) #Multilevel Picard object
solver3=ScaSML(equation=equation,net=solver1) #ScaSML object


# #run the test for NormalSphere
# test1=NormalSphere(equation,solver1,solver2,solver3)
# rhomax=test1.test(r"results_full_history/Neumann_Boundary/100d")
# #run the test for SimpleUniform
# test2=SimpleUniform(equation,solver1,solver2,solver3)
# test2.test(r"results_full_history/Neumann_Boundary/100d")
#run the test for ConvergenceRate
test3=ConvergenceRate(equation,solver1,solver2,solver3)
test3.test(r"results_full_history/Neumann_Boundary/100d")



#finish wandb
wandb.finish()
