# We will use this file to run the experiment. We will train the model and then test it on the NormalSphere test. We will also profile the test to see the time taken by each solver to solve the equation. We will save the profiler results and upload them to wandb.
import sys
import os
#add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# import the required libraries
from equations.equations import Grad_Dependent_Nonlinear
from optimizers.Adam import Adam 
# L_inf has been removed
from tests.SimpleUniform import SimpleUniform
from tests.ConvergenceRate import ConvergenceRate
from tests.InferenceScaling import InferenceScaling
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
#set default float to float16
dde.config.set_default_float("float16")
# fix random seed for jax
jax.random.PRNGKey(0)
# device configuration
device = jax.default_backend()
print(device)
if device == 'gpu':
    # get PINNU name
    gpu_name = jax.devices()[0].device_kind

#initialize wandb
wandb.init(project="Grad_Dependent_Nonlinear", notes="20 d", tags=["Adam training","L_inf_training"],mode="disabled") #debug mode
# wandb.init(project="Grad_Dependent_Nonlinear", notes="20 d", tags=["Adam training","L_inf_training"]) #working mode
wandb.config.update({"device": device}) # record device type

#initialize the equation
equation=Grad_Dependent_Nonlinear(n_input=21,n_output=1)
#check if trained model is already saved
if os.path.exists(r"results/Grad_Dependent_Nonlinear/20d/model.ckpt-?"):
    '''To Do: Retrain the model with new data points& Try new methods to reduce errors'''
    #load the model
    net=dde.maps.jax.FNN([21]+[50]*5+[1], "tanh", "Glorot normal")
    terminal_transform = equation.terminal_transform
    net.apply_output_transform(terminal_transform)
    data = equation.generate_data()
    model = dde.Model(data,net)
    model.restore(r"results/Grad_Dependent_Nonlinear/20d/model.ckpt-?",verbose=1)
    # set is_train to False
    is_train = False
else:
    #initialize the FNN
    #same layer width
    net1=dde.maps.jax.FNN([21]+[50]*5+[1], "tanh", "Glorot normal")
    net2=dde.maps.jax.FNN([21]+[50]*5+[1], "tanh", "Glorot normal")
    data1 = equation.generate_data()
    data2 = equation.generate_data()
    model1 = dde.Model(data1,net1)
    model2 = dde.Model(data2,net2)
    is_train = True


#initialize the normal sphere test
solver1_1= model1 #PINN network
solver1_2 = model2 #PINN network
solver2=MLP(equation=equation) #Multilevel Picard object
solver3_1=ScaSML(equation=equation,PINN=solver1_1) #ScaSML object
solver3_2=ScaSML(equation=equation,PINN=solver1_2) #ScaSML object


#run the test for SimpleUniform
test2=SimpleUniform(equation,solver1_1,solver2,solver3_1,is_train)
test2.test(r"results/Grad_Dependent_Nonlinear/20d")
# #run the test for ConvergenceRate
# test3=ConvergenceRate(equation,solver1_2,solver2,solver3_2, is_train)
# test3.test(r"results/Grad_Dependent_Nonlinear/20d")


#finish wandb
wandb.finish()
