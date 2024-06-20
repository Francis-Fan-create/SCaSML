from torch.optim import Adam, LBFGS
import deepxde as dde
import wandb
import torch
from torch.optim.lr_scheduler import ExponentialLR  

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

    def train(self,save_path,cycle=14,adam_every=500,lbfgs_every=10,metrics=["l2 relative error","mse"]):
        #interleaved training of adam and lbfgs
        loss_weights=[1e-3]*(self.n_input-1)+[1]+[1e-2]
        wandb.config.update({"cycle": cycle, "adam_every": adam_every, "lbfgs_every": lbfgs_every,"loss_weights":loss_weights}) # record hyperparameters
        adam=self.Adam()
        lbfgs=self.LBFGS()
        for i in range(cycle):
            self.model.compile(optimizer=adam,metrics=metrics,loss_weights=loss_weights)
            self.model.train(iterations=adam_every, display_every=10)
            # log a list of Adam losses and metrics, which are both lists, one by one
            counter1=0
            for loss in self.model.train_state.loss_train:
                counter1+=1
                wandb.log({"Adam loss_{:d}".format(counter1): loss})
            counter2=0
            for metric in self.model.train_state.metrics_test:
                counter2+=1
                wandb.log({"Adam metric_{:d}".format(counter2): metric})

            self.model.compile(optimizer=lbfgs,metrics=metrics,loss_weights=loss_weights)
            self.model.train(iterations=lbfgs_every, display_every=1)
            counter3=0
            for loss in self.model.train_state.loss_train:
                counter3+=1
                wandb.log({"LBFGS loss_{:d}".format(counter3): loss})
            counter4=0
            for metric in self.model.train_state.metrics_test:
                counter4+=1
                wandb.log({"LBFGS metric_{:d}".format(counter4): metric})
        #stablize the training by further training with Adam
        self.model.compile(optimizer=adam,metrics=metrics,loss_weights=loss_weights)
        self.model.train(iterations=40*adam_every, display_every=10)
        # log a list of Adam losses and metrics, which are both lists, one by one
        counter5=0
        for loss in self.model.train_state.loss_train:
            counter5+=1
            wandb.log({"Adam loss_{:d}".format(counter5): loss})
        counter6=0
        for metric in self.model.train_state.metrics_test:
            counter6+=1
            wandb.log({"Adam metric_{:d}".format(counter6): metric})
        #save the model
        torch.save(self.net.state_dict(), save_path)
        #log the model
        wandb.log_model(path=save_path, name="model")
        return self.model