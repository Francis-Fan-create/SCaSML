from torch.optim import Adam, LBFGS
import deepxde as dde
import wandb
import torch
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR

class L_inf(object):
    '''L_inf optimizer class for optimizing neural networks in the context of solving differential equations.
    
    Attributes:
        net (torch.nn.Module): The neural network model.
        data (dde.data): The data object containing training and boundary data.
        n_input (int): Number of input features.
        n_output (int): Number of output features.
        model (dde.Model): The DeepXDE model object.
        equation (object): The differential equation object.
        geom (object): The geometry associated with the differential equation.
    '''
    def __init__(self, n_input, n_output, net, data, equation):
        '''Initializes the L_inf optimizer with the necessary parameters and model.
        
        Args:
            n_input (int): Number of input features.
            n_output (int): Number of output features.
            net (torch.nn.Module): The neural network model.
            data (dde.data): The data object containing training and boundary data.
            equation (object): The differential equation object to be solved.
        '''
        self.net = net
        self.data = data
        self.n_input = n_input
        self.n_output = n_output
        self.model = dde.Model(data, net)
        self.equation = equation
        self.geom = equation.geometry()

    def Adam(self, lr=1e-2, weight_decay=1e-4, gamma=0.9):
        '''Configures the Adam optimizer with exponential learning rate decay.
        
        Args:
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 penalty).
            gamma (float): Multiplicative factor of learning rate decay.
        
        Returns:
            torch.optim.Adam: Configured Adam optimizer.
        '''
        adam = Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        wandb.config.update({"Adam lr": lr, "Adam weight_decay": weight_decay, "Adam gamma": gamma})
        return adam

    def LBFGS(self, lr=1e-2, max_iter=1000, tolerance_change=1e-5, tolerance_grad=1e-3):
        '''Configures the LBFGS optimizer.
        
        Args:
            lr (float): Learning rate.
            max_iter (int): Maximum number of iterations.
            tolerance_change (float): Termination tolerance on function value/parameter changes.
            tolerance_grad (float): Termination tolerance on gradient.
        
        Returns:
            torch.optim.LBFGS: Configured LBFGS optimizer.
        '''
        lbfgs = LBFGS(self.net.parameters(), lr=lr, max_iter=max_iter, tolerance_change=tolerance_change, tolerance_grad=tolerance_grad)
        wandb.config.update({"LBFGS lr": lr, "LBFGS max_iter": max_iter, "LBFGS tolerance_change": tolerance_change, "LBFGS tolerance_grad": tolerance_grad})
        return lbfgs
    
    def get_anchors(self, domain_anchors, boundary_anchors, refinement_num=20):
        '''Generates anchor points using the projection gradient method.
        
        Args:
            domain_anchors (int): Number of domain anchor points.
            boundary_anchors (int): Number of boundary anchor points.
            refinement_num (int): Number of refinement steps for anchor points.
        
        Returns:
            tuple: A tuple containing two numpy arrays of domain and boundary anchor points, respectively.
                   The shape of each array is (N, D) where N is the number of points and D is the dimension.
        '''
        geom = self.geom
        eta = 1 / refinement_num
        domain_points = geom.random_points(domain_anchors)
        boundary_points = geom.random_boundary_points(boundary_anchors)
        tensor_domain_points = torch.tensor(domain_points, requires_grad=True)
        tensor_boundary_points = torch.tensor(boundary_points, requires_grad=True)
        net = self.net
        net.eval()
        eq = self.equation
        for i in range(refinement_num):
            tensor_domain_points.requires_grad = True
            prediction_domain = net(tensor_domain_points)
            loss_domain = torch.mean(eq.PDE_loss(tensor_domain_points, prediction_domain, torch.autograd.grad(prediction_domain, tensor_domain_points, grad_outputs=torch.ones_like(prediction_domain), create_graph=True, retain_graph=True)[0]))
            grad_domain = torch.autograd.grad(loss_domain, tensor_domain_points)[0]
            tensor_domain_points = tensor_domain_points.detach() + eta * torch.sign(grad_domain.detach())
            tensor_domain_points[:, -1] = torch.clamp(tensor_domain_points[:, -1], eq.t0, eq.T)
            tensor_boundary_points.requires_grad = True
            prediction_boundary = net(tensor_boundary_points)
            loss_boundary = torch.mean((prediction_boundary - torch.tensor(eq.terminal_constraint(boundary_points), requires_grad=True)) ** 2)
            grad_boundary = torch.autograd.grad(loss_boundary, tensor_boundary_points)[0]
            tensor_boundary_points = tensor_boundary_points.detach() + eta * torch.sign(grad_boundary.detach())
            tensor_boundary_points[:, -1] = torch.clamp(tensor_boundary_points[:, -1], eq.t0, eq.T)
        return tensor_domain_points.detach().cpu().numpy(), tensor_boundary_points.detach().cpu().numpy()
    
    def train(self, save_path, domain_anchors=100, boundary_anchors=100, adam_iterations=5000, metrics=["l2 relative error", "mse"]):
        '''Trains the model using an interleaved training strategy of Adam and LBFGS optimizers.
        
        Args:
            save_path (str): Path to save the trained model.
            domain_anchors (int): Number of domain anchor points.
            boundary_anchors (int): Number of boundary anchor points.
            adam_iterations (int): Number of iterations for Adam optimizer .
            metrics (list): List of metrics to evaluate during training.
        
        Returns:
            dde.Model: The trained model.
        '''
        loss_weights = [1e-6] * (self.n_input - 1) + [7e-4] + [7e-4]
        wandb.config.update({ "adam_iterations": adam_iterations, "loss_weights": loss_weights})
        adam = self.Adam()
        data = self.data
        domain_points, boundary_points = self.get_anchors(domain_anchors, boundary_anchors)
        data.replace_with_anchors(domain_points)
        data.train_x_bc = boundary_points
        self.model.compile(optimizer=adam, metrics=metrics, loss_weights=loss_weights)
        self.model.train(iterations=adam_iterations, display_every=10)
        counter1 = 0
        for loss in self.model.train_state.loss_train:
            counter1 += 1
            wandb.log({"Adam loss_{:d}".format(counter1): loss})
        counter2 = 0
        for metric in self.model.train_state.metrics_test:
            counter2 += 1
            wandb.log({"Adam metric_{:d}".format(counter2): metric})
        torch.save(self.net.state_dict(), save_path)
        wandb.log_model(path=save_path, name="model")
        return self.model