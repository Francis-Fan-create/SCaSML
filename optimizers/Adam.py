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

    def train(self, save_path, iters = 5000, metrics=["l2 relative error", "mse"]):
        '''Trains the model using an interleaved strategy of Adam optimizer.
        
        Args:
            save_path (str): Path to save the trained model.
            iters (int): Number of iterations for training.
        
        Returns:
            dde.Model: The trained model.
        '''
        # # Stabilize the training by further training with Adam
        # self.model.compile("adam", lr=1e-3, metrics=metrics)
        # # Deepxde does not implement Model.save() for jax
        # self.model.train(iterations=iters, display_every=10, disregard_previous_best= True)
        # RAR training
        geom = self.equation.geometry()
        # Use Adaptive Refinement for training
        data_pool = geom.random_points(20* iters)
        err = 1.0
        for i in range(4):
            residual = self.model.predict(data_pool, operator=self.equation.PDE_loss)
            err_array = jnp.abs(residual)
            err = jnp.mean(err_array)
            print(f"Mean residual: {err}")
            train_id = jnp.argsort(err_array, stable=True)[:1000]
            train_data = data_pool[train_id,:][:,0,:]
            self.model.data.add_anchors(train_data)
            early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
            self.model.compile("adam", lr=1e-3, metrics=metrics)
            loss_history, train_state = self.model.train(iterations=iter//10, display_every=10, disregard_previous_best=True, callbacks=[early_stopping])
        dde.saveplot(loss_history, train_state, issave=True, isplot=True,output_dir=save_path)
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