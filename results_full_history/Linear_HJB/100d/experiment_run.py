# We will use this file to run the experiment. We will train the model and then test it on the NormalSphere test. We will also profile the test to see the time taken by each solver to solve the equation. We will save the profiler results and upload them to wandb.
import sys
import os
#add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# import the required libraries
from equations.equations import Linear_HJB
from optimizers.Adam import Adam 
# L_inf has been removed
from tests.NormalSphere import NormalSphere
from tests.SimpleUniform import SimpleUniform
from tests.ConvergenceRate import ConvergenceRate
from solvers.MLP import MLP
from solvers.ScaSML import ScaSML
import numpy as np
import torch
import wandb
import deepxde as dde
import jax


#fix random seed for dde
dde.config.set_random_seed(1234)
#use jax backend
dde.backend.set_default_backend('jax')
# fix random seed for jax
jax.random.PRNGKey(0)
# device configuration
device = jax.default_backend()
print(device)
if device == 'gpu':
    # get PINNU name
    gpu_name = jax.devices()[0].device_kind

#initialize wandb
wandb.init(project="Linear_HJB", notes="100 d", tags=["Adam training","L_inf_training"],mode="disabled") #debug mode
# wandb.init(project="Linear_HJB", notes="100 d", tags=["Adam training","L_inf_training"]) #working mode
wandb.config.update({"device": device}) # record device type

#initialize the equation
equation=Linear_HJB(n_input=101,n_output=1)
#check if trained model is already saved
if os.path.exists(r"results_full_history/Linear_HJB/100d/model.ckpt-?"):
    '''To Do: Retrain the model with new data points& Try new methods to reduce errors'''
    #load the model
    net=dde.maps.jax.FNN([101]+[50]*5+[1], "relu", "Glorot normal")
    data = equation.generate_data()
    model = dde.Model(data,net)
    model.restore(r"results_full_history/Linear_HJB/100d/model.ckpt-?",verbose=1)
    # set is_train to False
    is_train = False
else:
    #initialize the FNN
    #same layer width
    net=dde.maps.jax.FNN([101]+[50]*5+[1], "relu", "Glorot normal")
    data = equation.generate_data()
    model = dde.Model(data,net)
    is_train = True


#initialize the normal sphere test
solver1 = model #PINN network
solver2=MLP(equation=equation) #Multilevel Picard object
solver3=ScaSML(equation=equation,PINN=solver1) #ScaSML object


# #run the test for NormalSphere
# test1=NormalSphere(equation,solver1,solver2,solver3, is_train)
# rhomax=test1.test(r"results_full_history/Linear_HJB/100d")
#run the test for SimpleUniform
test2=SimpleUniform(equation,solver1,solver2,solver3,is_train)
test2.test(r"results_full_history/Linear_HJB/100d")
#run the test for ConvergenceRate
test3=ConvergenceRate(equation,solver1,solver2,solver3, is_train)
test3.test(r"results_full_history/Linear_HJB/100d")


#finish wandb
wandb.finish()
