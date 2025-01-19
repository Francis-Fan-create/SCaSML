import deepxde as dde
import wandb
import torch
import numpy as np
import optax
import jax.numpy as jnp
import jax

class L_inf(object):
    '''L_inf optimizer class for optimizing neural networks in the context of solving differential equations.
    
    Attributes:
        data (dde.data): The data object containing training and boundary data.
        n_input (int): Number of input features.
        n_output (int): Number of output features.
        model (dde.Model): The DeepXDE model object.
        equation (object): The differential equation object.
        geom (object): The geometry associated with the differential equation.
    '''
    def __init__(self, n_input, n_output, model, data, equation):
        '''Initializes the L_inf optimizer with the necessary parameters and model.
        
        Args:
            n_input (int): Number of input features.
            n_output (int): Number of output features.
            model (dde.Model): The DeepXDE model object.
            data (dde.data): The data object containing training and boundary data.
            equation (object): The differential equation object to be solved.
        '''
        self.data = data
        self.n_input = n_input
        self.n_output = n_output
        self.model = model
        self.equation = equation
        self.geom = equation.geometry()
    
    def get_anchors(self, domain_anchors, boundary_anchors, refinement_num=20):
        '''Generates anchor points using a gradient-based refinement (converted to JAX).
        
        Args:
            domain_anchors (int): Number of domain anchor points.
            boundary_anchors (int): Number of boundary anchor points.
            refinement_num (int): Number of refinement steps.
        
        Returns:
            tuple: Two arrays of domain and boundary anchor points.
        '''
        geom = self.geom
        eta = 1 / refinement_num
        domain_points = geom.random_points(domain_anchors)
        boundary_points = geom.random_boundary_points(boundary_anchors)
        eq = self.equation
        model = self.model
        # Placeholder approach: demonstrate how to move domain_points with sign of gradient
        # In practice, you'd construct a jax.grad-compatible PDE_loss
        for _ in range(refinement_num):
            preds_domain = model.predict(domain_points)
            # PDE gradient for domain
            def domain_loss_fn(pts):
                temp_PDE_loss = lambda points:eq.PDE_loss(points,preds_domain)
                return jnp.mean(temp_PDE_loss(pts)**2)
            
            grad_domain = jax.grad(domain_loss_fn)(domain_points)
            domain_points = domain_points + eta * jnp.sign(grad_domain)
            domain_points = domain_points.at[:, -1].set(
                jnp.clip(domain_points[:, -1], eq.t0, eq.T)
            )
            # PDE gradient for boundary
            preds_boundary = model.predict(boundary_points)
            def boundary_loss_fn(pts):
                cons = jnp.array(eq.terminal_constraint(pts))
                return jnp.mean((preds_boundary - cons) ** 2)
            grad_boundary = jax.grad(boundary_loss_fn)(boundary_points)
            boundary_points = boundary_points + eta * jnp.sign(grad_boundary)
            boundary_points = boundary_points.at[:, -1].set(
                jnp.clip(boundary_points[:, -1], eq.t0, eq.T)
            )
        return domain_points, boundary_points
    
    def train(self, save_path, cycle=2, domain_anchors=100, boundary_anchors=100, adam_every=100, metrics=["l2 relative error", "mse"]):
        '''Trains the model using an interleaved training strategy of Adam and LBFGS optimizers.
        
        Args:
            save_path (str): Path to save the trained model.
            cycle(int): Number of cycles for training.
            domain_anchors (int): Number of domain anchor points.
            boundary_anchors (int): Number of boundary anchor points.
            adam_every (int): Number of iterations for Adam optimizer in each cycle.
            metrics (list): List of metrics to evaluate during training.
        
        Returns:
            dde.Model: The trained model.
        '''
        eq = self.equation
        if eq.__class__.__name__ == "Linear_HJB" or "Grad_Dependent_Nonlinear":
            loss_weights = [1] + [1e-2]
        elif eq.__class__.__name__ == "Neumann_Boundary":
            loss_weights = [1] + [1e-2]*2
        wandb.config.update({"adam_iterations": adam_every, "loss_weights": loss_weights})
        data = self.data
        for i in range(cycle):
            print("Cycle: ", i)
            domain_points, boundary_points = self.get_anchors(domain_anchors, boundary_anchors)
            data.replace_with_anchors(domain_points)
            data.train_x_bc = boundary_points
            # Use compiled model with Adam
            self.model.compile("adam", lr=1e-2, metrics=metrics, loss_weights=loss_weights)
            # This .train call is assumed to be jax-friendly in the new dde version
            loss_history, train_state= self.model.train(iterations=adam_every, display_every=10)
            dde.saveplot(loss_history, train_state, issave=True, isplot=True,output_dir=save_path)
            for idx, loss_val in enumerate(self.model.train_state.loss_train, start=1):
                wandb.log({f"Adam loss_{idx}": loss_val})
            for idx, metric_val in enumerate(self.model.train_state.metrics_test, start=1):
                wandb.log({f"Adam metric_{idx}": metric_val})
        return self.model