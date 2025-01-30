import deepxde as dde
import wandb
import optax 
import jax.numpy as jnp
from jaxopt import LBFGS
from solvers.ScaSML_full_history import ScaSML_full_history


class Adam(object):
    '''Adam optimizer.
    
    This class combines the Adam and LBFGS optimizers for training neural networks. It is specifically designed for use with the deepxde framework to solve differential equations using deep learning.
    
    Attributes:
        data (dde.data.Data): The dataset used for training the model.
        n_input (int): Number of input features.
        n_output (int): Number of output features.
        model (dde.Model): The deepxde model that wraps around the neural network and the dataset.
    '''
    def __init__(self, n_input, n_output, model, data, equation):
        '''Initializes the Adam optimizer with the network, data, and dimensions.
        
        Args:
            n_input (int): Number of input features.
            n_output (int): Number of output features.
            model (dde.Model): The DeepXDE model object.
            data (dde.data.Data): The dataset used for training the model.
            equation (object): The differential equation object to be solved.
        '''
        # Initialize the optimizer parameters
        self.model = model
        self.data = data
        self.n_input = n_input
        self.n_output = n_output
        self.equation = equation
        # We do not need to initialize wandb here, as it is already initialized in the main script

    def pretrain(self, save_path, iters = 5000, metrics=["l2 relative error", "mse"]):
        '''Trains the model using an interleaved strategy of Adam optimizer.
        
        Args:
            save_path (str): Path to save the trained model.
            iters (int): Number of iterations for training.
        
        Returns:
            dde.Model: The trained model.
        '''
        # Stabilize the training by further training with Adam
        self.model.compile("adam", lr=1e-2, metrics=metrics, decay=("inverse time", 1000, 0.3))
        # Deepxde does not implement Model.save() for jax
        loss_history, train_state= self.model.train(iterations=iters, display_every=10, disregard_previous_best= True)
        dde.saveplot(loss_history, train_state, issave=True, isplot=True,output_dir=save_path+"/pretrain")
        # Log a list of Adam losses and metrics, which are both lists, one by one
        counter1 = 0
        for loss in self.model.train_state.loss_train:
            counter1 += 1
            wandb.log({"Adam loss_{:d}".format(counter1): loss})
        counter2 = 0
        for metric in self.model.train_state.metrics_test:
            counter2 += 1
            wandb.log({"Adam metric_{:d}".format(counter2): metric})
        # Log the model
        # wandb.log_model(path=save_path, name="model")
        return self.model
    
    def train(self, save_path, iters = 5000, metrics=["l2 relative error", "mse"]):
        '''Trains the model and uses fine-tuning on inference.
        
        Args:
            save_path (str): Path to save the trained model.
            iters (int): Number of iterations for training.
        
        Returns:
            dde.Model: The trained model.
        '''
        model = self.train(save_path, iters, metrics)
        inferencer = ScaSML_full_history(equation= self.equation, PINN = model)
        geom = self.equation.geometry()
        train_data = geom.random_points(100)
        infer_func = lambda x_t: inferencer.uz_solve(2, None, x_t)[:, 0][:, jnp.newaxis]
        target = jnp.zeros((train_data.shape[0], 1))
        inference_condition = dde.PointSetOperatorBC(train_data, target, infer_func)
        finetune_data = dde.data.TimePDE(
                                geom, # Geometry of the domain.
                                self.equation.PDE_loss, # PDE loss function.
                                [inference_condition], # Additional conditions.
                                num_domain=100, # Number of domain points.
                                num_boundary=0, # Number of boundary points.
                                num_initial=0,  # Number of initial points.
                                solution=self.exact_solution   # Incorporates exact solution for error metrics.
                            )
        model.data = finetune_data
        model.compile("adam", lr=1e-3, metrics=metrics, decay=("inverse time", 1000, 0.3))
        loss_history, train_state= model.train(iterations=iters, display_every=10, disregard_previous_best= True)
        dde.saveplot(loss_history, train_state, issave=True, isplot=True,output_dir=save_path+"/finetune")
        # Log a list of Adam losses and metrics, which are both lists, one by one
        counter1 = 0
        for loss in self.model.train_state.loss_train:
            counter1 += 1
            wandb.log({"Adam loss_{:d}".format(counter1): loss})
        counter2 = 0
        for metric in self.model.train_state.metrics_test:
            counter2 += 1
            wandb.log({"Adam metric_{:d}".format(counter2): metric})
        # Log the model
        # wandb.log_model(path=save_path, name="model")
        return self.model