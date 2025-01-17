# We will use this file to run the experiment. We will train the model and then test it on the NormalSphere test. We will also profile the test to see the time taken by each solver to solve the equation. We will save the profiler results and upload them to wandb.
import sys
import os
#add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# import the required libraries
from equations.equations import Linear_HJB
from models.FNN import FNN
from optimizers.Adam_LBFGS import Adam_LBFGS 
from optimizers.L_inf import L_inf
from tests.NormalSphere import NormalSphere
from tests.SimpleUniform import SimpleUniform
from tests.ConvergenceRate import ConvergenceRate
from solvers.MLP import MLP
from solvers.ScaSML import ScaSML
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
    # get PINNU name
    gpu_name = torch.cuda.get_device_name()

#initialize wandb
wandb.init(project="Linear_HJB", notes="100 d", tags=["Adam_LBFGS training","L_inf_training"],mode="disabled") #debug mode
# wandb.init(project="Linear_HJB", notes="100 d", tags=["Adam_LBFGS training","L_inf_training"]) #working mode
wandb.config.update({"device": device.type}) # record device type

#initialize the equation
equation=Linear_HJB(n_input=101,n_output=1)
#check if trained model is already saved
if os.path.exists(r"results/Linear_HJB/100d/model_weights_L_inf.params"):
    '''To Do: Retrain the model with new data points& Try new methods to reduce errors'''
    #load the model
    net=FNN([101]+[50]*5+[1],equation)
    net.load_state_dict(torch.load(r"results/Linear_HJB/100d/model_weights_L_inf.params",map_location=device)) #the other indexes are left for external resources of weights
    trained_net=net
    is_train = False
else:
    #initialize the FNN
    layers=[101]+[50]*5+[1]
    net=FNN(layers,equation)
    is_train = True


#initialize the normal sphere test
solver1= net #PINN network
solver2=MLP(equation=equation) #Multilevel Picard object
solver3=ScaSML(equation=equation,PINN=solver1) #ScaSML object


#run the test for NormalSphere
test1=NormalSphere(equation,solver1,solver2,solver3, is_train)
rhomax=test1.test(r"results/Linear_HJB/100d")
#run the test for SimpleUniform
test2=SimpleUniform(equation,solver1,solver2,solver3,is_train)
test2.test(r"results/Linear_HJB/100d")
#run the test for ConvergenceRate
test3=ConvergenceRate(equation,solver1,solver2,solver3, is_train)
test3.test(r"results/Linear_HJB/100d")


#finish wandb
wandb.finish()
