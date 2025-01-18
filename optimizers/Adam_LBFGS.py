import deepxde as dde
import wandb
import optax 
import jax.numpy as jnp
from jaxopt import LBFGS


class Adam_LBFGS(object):
    '''Adam-LBFGS optimizer.
    
    This class combines the Adam and LBFGS optimizers for training neural networks. It is specifically designed for use with the deepxde framework to solve differential equations using deep learning.
    
    Attributes:
        net (object): The neural network model to be optimized.
        data (dde.data.Data): The dataset used for training the model.
        n_input (int): Number of input features.
        n_output (int): Number of output features.
        model (dde.Model): The deepxde model that wraps around the neural network and the dataset.
    '''
    def __init__(self, n_input, n_output, net, data, equation):
        '''Initializes the Adam_LBFGS optimizer with the network, data, and dimensions.
        
        Args:
            n_input (int): Number of input features.
            n_output (int): Number of output features.
            net (object): The neural network model to be optimized.
            data (dde.data.Data): The dataset used for training the model.
            equation (object): The differential equation object to be solved.
        '''
        # Initialize the optimizer parameters
        self.net = net
        self.data = data
        self.n_input = n_input
        self.n_output = n_output
        self.model = dde.Model(data, net)
        self.equation = equation
        # We do not need to initialize wandb here, as it is already initialized in the main script

    def Adam(self, lr=1e-2, weight_decay=1e-4, gamma=0.9):
        '''
        Initializes and returns an Adam optimizer with an exponential learning rate scheduler.
        ...
        '''
        # adam = Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = ExponentialLR(adam, gamma=gamma)
        adam = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.adam(lr)
        )
        wandb.config.update({"Adam lr": lr, "Adam weight_decay": weight_decay, "Adam gamma": gamma})
        return adam

    def LBFGS(self, lr=1e-2, max_iter=1000, tolerance_change=1e-5, tolerance_grad=1e-3):
        '''
        Initializes and returns an LBFGS optimizer.
        ...
        '''
        # lbfgs = LBFGS(self.net.parameters(), lr=lr, max_iter=max_iter, tolerance_change=tolerance_change, tolerance_grad=tolerance_grad)
        lbfgs = LBFGS(fun=None, maxiter=max_iter, tol=tolerance_grad)  # Placeholder usage for JAX-based LBFGS
        wandb.config.update({"LBFGS lr": lr, "LBFGS max_iter": max_iter, "LBFGS tolerance_change": tolerance_change, "LBFGS tolerance_grad": tolerance_grad})
        return lbfgs

    def train(self, save_path, cycle=4, adam_every=100, lbfgs_every=10, metrics=["l2 relative error", "mse"]):
        '''Trains the model using an interleaved strategy of Adam and LBFGS optimizers.
        
        Args:
            save_path (str): Path to save the trained model.
            cycle (int): Number of cycles of interleaved training.
            adam_every (int): Number of iterations to run Adam optimizer in each cycle.
            lbfgs_every (int): Number of iterations to run LBFGS optimizer in each cycle.
            metrics (list of str): List of metrics to evaluate during training.
        
        Returns:
            dde.Model: The trained model.
        '''
        eq= self.equation
        # Interleaved training of Adam and LBFGS
        if eq.__class__.__name__ == "Linear_HJB" or "Grad_Dependent_Nonlinear":
            loss_weights = [1e-3] * (self.n_input - 1) + [1] + [1e-2] 
        elif eq.__class__.__name__ == "Neumann_Boundary":
            loss_weights = [1e-3] * (self.n_input - 1) + [1] + [1e-2]*2
        wandb.config.update({"cycle": cycle, "adam_every": adam_every, "lbfgs_every": lbfgs_every, "loss_weights": loss_weights})  # Record hyperparameters
        adam = self.Adam()
        lbfgs = self.LBFGS()
        try:
            for i in range(cycle):
                print(f"Cycle:{i}")
                self.model.compile(optimizer=adam, metrics=metrics, loss_weights=loss_weights)
                self.model.train(iterations=adam_every, display_every=10)
                # Log a list of Adam losses and metrics, which are both lists, one by one
                counter1 = 0
                for loss in self.model.train_state.loss_train:
                    counter1 += 1
                    wandb.log({"Adam loss_{:d}".format(counter1): loss})
                counter2 = 0
                for metric in self.model.train_state.metrics_test:
                    counter2 += 1
                    wandb.log({"Adam metric_{:d}".format(counter2): metric})

                self.model.compile(optimizer=lbfgs, metrics=metrics, loss_weights=loss_weights)
                self.model.train(iterations=lbfgs_every, display_every=1)
                counter3 = 0
                for loss in self.model.train_state.loss_train:
                    counter3 += 1
                    wandb.log({"LBFGS loss_{:d}".format(counter3): loss})
                counter4 = 0
                for metric in self.model.train_state.metrics_test:
                    counter4 += 1
                    wandb.log({"LBFGS metric_{:d}".format(counter4): metric})
            # Stabilize the training by further training with Adam
            self.model.compile(optimizer=adam, metrics=metrics, loss_weights=loss_weights)
            _ , train_state = self.model.train(iterations=5 * adam_every, display_every=10)
            # Log a list of Adam losses and metrics, which are both lists, one by one
            counter5 = 0
            for loss in self.model.train_state.loss_train:
                counter5 += 1
                wandb.log({"Adam loss_{:d}".format(counter5): loss})
            counter6 = 0
            for metric in self.model.train_state.metrics_test:
                counter6 += 1
                wandb.log({"Adam metric_{:d}".format(counter6): metric})
            # Save the model
            dde.saveplot(_, train_state, issave=True, isplot=False, output_dir=save_path)
            # Log the model
            # wandb.log_model(path=save_path, name="model")
            return self.model
        except KeyboardInterrupt:
            # save the model
            dde.saveplot(_, train_state, issave=True, isplot=False, output_dir=save_path)
            # Log the model
            # wandb.log_model(path=save_path, name="model")
            return self.model