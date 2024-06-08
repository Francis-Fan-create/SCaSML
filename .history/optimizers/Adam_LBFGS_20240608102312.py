from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ExponentialLR
import deepxde as dde
import wandb
import torch

class Adam_LBFGS(object):
    '''Adam-LBFGS optimizer'''
    def __init__(self,n_input,n_output,net,data):
        #initialize the optimizer parameters
        self.net=net
        self.data=data
        self.n_input=n_input
        self.n_output=n_output
        self.model=dde.Model(data,net)
        #we do not need to initialize wandb here, as it is already initialized in the main script

    def Adam(self, lr=1e-2,weight_decay=1e-4,gamma=0.9):
        #Adam optimizer
        adam = Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ExponentialLR(adam, gamma=gamma)
        wandb.config.update({"Adam lr": lr, "Adam weight_decay": weight_decay, "Adam gamma": gamma})  # record hyperparameters
        return adam

    def LBFGS(self, lr=1e-2,max_iter=1000,tolerance_change=1e-5,tolerance_grad=1e-3):
        #LBFGS optimizer
        lbfgs = LBFGS(self.net.parameters(), lr=lr, max_iter=max_iter, tolerance_change=tolerance_change, tolerance_grad=tolerance_grad)
        wandb.config.update({"LBFGS lr": lr, "LBFGS max_iter": max_iter, "LBFGS tolerance_change": tolerance_change, "LBFGS tolerance_grad": tolerance_grad})  # record hyperparameters
        return lbfgs

    def train(self,save_path,cycle=40,adam_every=500,lbfgs_every=10,metrics=["l2 relative error","mse"]):
        #interleaved training of adam and lbfgs
        wandb.config.update({"cycle": cycle, "adam_every": adam_every, "lbfgs_every": lbfgs_every}) # record hyperparameters
        for i in range(cycle):
            if self.model.train_state.epoch>0:
                loss_weights=torch.softmax(torch.tensor(self.model.train_state.loss_train)).detach().tolist()                           #use self adaptive loss weights
            else:
                loss_weights=[1]*len(metrics)
            wandb.log({"loss_weights": loss_weights}) 
            self.model.compile(optimizer=self.Adam(),metrics=metrics,loss_weights=loss_weights)
            self.model.train(epochs=adam_every, display_every=10, metrics=metrics)
            wandb.log({"Adam loss": self.model.train_state.loss_train, "Adam metrics": self.model.train_state.metrics_test})  # record loss and metrics

            self.model.compile(optimizer=self.LBFGS(),metrics=metrics,loss_weights=loss_weights)
            self.model.train(epochs=lbfgs_every, display_every=1, metrics=metrics)
            wandb.log({"LBFGS loss": self.model.train_state.loss_train, "LBFGS metrics": self.model.train_state.metrics_test})  # record loss and metrics

            gpu_memory=torch.cuda.memory_allocated()
            gpu_memory_max=torch.cuda.max_memory_allocated()
            wandb.log({"GPU memory": gpu_memory, "GPU memory max": gpu_memory_max})
        self.model.save(save_path)
        return self.model