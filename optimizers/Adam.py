import deepxde as dde
import wandb
import optax 
import jax.numpy as jnp

class Adam(object):
    '''Adam optimizer.
    
    This class combines the Adam and LBFGS optimizers for training neural networks. It is specifically designed for use with the deepxde framework to solve differential equations using deep learning.
    
    Attributes:
        data (dde.data.Data): The dataset used for training the model.
        n_input (int): Number of input features.
        n_output (int): Number of output features.
        model (dde.Model): The deepxde model that wraps around the neural network and the dataset.
    '''
    def __init__(self, n_input, n_output, model, equation):
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
        self.n_input = n_input
        self.n_output = n_output
        self.equation = equation
        # We do not need to initialize wandb here, as it is already initialized in the main script

    def train(self, save_path, iters = None, metrics=["l2 relative error", "mse"]):
        '''Trains the model using an interleaved strategy of Adam optimizer.
        
        Args:
            save_path (str): Path to save the trained model.
            iters (int): Number of iterations for training.
        
        Returns:
            dde.Model: The trained model.
        '''
        real_lr = self.equation.lr
        # Stabilize the training by further training with Adam
        self.model.compile("adam", lr=real_lr, metrics=metrics)
        if iters is not None:
            real_iters = iters
        else:
            real_iters = self.equation.iters
        # Deepxde does not implement Model.save() for jax
        loss_history, train_state = self.model.train(iterations=real_iters, display_every=10, disregard_previous_best= True)
        # dde.saveplot(loss_history, train_state, issave=True, isplot=True,output_dir=save_path)
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