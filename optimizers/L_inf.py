from torch.optim import Adam, LBFGS
import deepxde as dde
import wandb
import torch
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR

class L_inf(object):
    ''':L_inf optimizer'''
    def __init__(self,n_input,n_output,net,data,equation):
        #initialize the optimizer parameters
        self.net=net
        self.data=data
        self.n_input=n_input
        self.n_output=n_output
        self.model=dde.Model(data,net)
        self.equation=equation
        self.geom=equation.geometry()
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
    
    def get_anchors(self,domain_anchors,boundary_anchors,refinement_num=20):
        #generate anchors from projection gradient method
        geom=self.geom
        eta=1/refinement_num
        domain_points=geom.random_points(domain_anchors)
        boundary_points=geom.random_boundary_points(boundary_anchors)
        tensor_domain_points=torch.tensor(domain_points,requires_grad=True)
        tensor_boundary_points=torch.tensor(boundary_points,requires_grad=True)
        net=self.net
        net.eval()
        eq=self.equation
        for i in range(refinement_num):
            tensor_domain_points.requires_grad=True
            prediction_domain=net(tensor_domain_points)
            loss_domain=torch.mean(eq.PDE_loss(tensor_domain_points,prediction_domain,torch.autograd.grad(prediction_domain,tensor_domain_points,grad_outputs=torch.ones_like(prediction_domain),create_graph=True,retain_graph=True)[0]))
            grad_domain=torch.autograd.grad(loss_domain,tensor_domain_points)[0]
            tensor_domain_points=tensor_domain_points.detach()+eta*torch.sign(grad_domain.detach())    
            tensor_domain_points[:,-1]=torch.clamp(tensor_domain_points[:,-1],eq.t0,eq.T)
            tensor_boundary_points.requires_grad=True
            prediction_boundary=net(tensor_boundary_points)
            loss_boundary=torch.mean((prediction_boundary-torch.tensor(eq.terminal_constraint(boundary_points),requires_grad=True))**2)
            grad_boundary=torch.autograd.grad(loss_boundary,tensor_boundary_points)[0]
            tensor_boundary_points=tensor_boundary_points.detach()+eta*torch.sign(grad_boundary.detach())
            tensor_boundary_points[:,-1]=torch.clamp(tensor_boundary_points[:,-1],eq.t0,eq.T)
        return tensor_domain_points.detach().cpu().numpy(),tensor_boundary_points.detach().cpu().numpy()
    
    def train(self,save_path,cycle=14,domain_anchors=2000,boundary_anchors=500,adam_every=50,lbfgs_every=10,metrics=["l2 relative error","mse"]):
        #interleaved training of adam and lbfgs
        loss_weights=[1e-3]*(self.n_input-1)+[1]+[1e-2]
        wandb.config.update({"cycle": cycle, "adam_every": adam_every, "lbfgs_every": lbfgs_every,"loss_weights":loss_weights}) # record hyperparameters
        adam=self.Adam()
        lbfgs=self.LBFGS()
        data=self.data
        for i in range(cycle):
            domain_points,boundary_points=self.get_anchors(domain_anchors,boundary_anchors)
            data.replace_with_anchors(domain_points)
            data.train_x_bc=boundary_points
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
        domain_points,boundary_points=self.get_anchors(domain_anchors,boundary_anchors)
        data.replace_with_anchors(domain_points)
        data.train_x_bc=boundary_points        
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