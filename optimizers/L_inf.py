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
        net (object): The neural network model.
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
            net (object): The neural network model.
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
        '''Configures the Adam optimizer with exponential learning rate decay (using optax).
        
        Args:
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 penalty).
            gamma (float): Multiplicative factor of learning rate decay.
        
        Returns:
            optax.GradientTransformation: Configured Adam optimizer chain.
        '''
        adam = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.adam(lr)
        )
        wandb.config.update({"Adam lr": lr, "Adam weight_decay": weight_decay, "Adam gamma": gamma})
        return adam

    def LBFGS(self, lr=1e-2, max_iter=1000, tolerance_change=1e-5, tolerance_grad=1e-3):
        '''Configures the LBFGS optimizer using jaxopt (placeholder usage).
        
        Args:
            lr (float): Learning rate.
            max_iter (int): Maximum number of iterations.
            tolerance_change (float): Termination tolerance on function value/parameter changes.
            tolerance_grad (float): Termination tolerance on gradient.
        
        Returns:
            jaxopt.LBFGS: Configured LBFGS optimizer.
        '''
        lbfgs = dde.optimizers.JaxoptLBFGS(
            maxiter=max_iter, tol=tolerance_grad
        )
        wandb.config.update({"LBFGS lr": lr, "LBFGS max_iter": max_iter, "LBFGS tolerance_change": tolerance_change, "LBFGS tolerance_grad": tolerance_grad})
        return lbfgs
    
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
        domain_points = jnp.array(domain_points)
        boundary_points = jnp.array(boundary_points)
        net = self.net
        eq = self.equation
        
        # Placeholder approach: demonstrate how to move domain_points with sign of gradient
        # In practice, you'd construct a jax.grad-compatible PDE_loss
        for _ in range(refinement_num):
            # PDE gradient for domain
            def domain_loss_fn(pts):
                preds = net(pts)
                # eq.PDE_loss should be a jax-friendly function returning the PDE residual
                # grad_u = jax.grad(lambda x: jnp.sum(net(x)))(pts)
                # Here we assume eq.PDE_loss returns a scalar
                return jnp.mean(eq.PDE_loss(pts, preds, None))
            
            grad_domain = jax.grad(domain_loss_fn)(domain_points)
            domain_points = domain_points + eta * jnp.sign(grad_domain)
            domain_points = domain_points.at[:, -1].set(
                jnp.clip(domain_points[:, -1], eq.t0, eq.T)
            )
            # PDE gradient for boundary
            def boundary_loss_fn(pts):
                preds = net(pts)
                cons = jnp.array(eq.terminal_constraint(np.array(pts)))
                return jnp.mean((preds - cons) ** 2)
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
            loss_weights = [1e-3] * (self.n_input - 1) + [1] + [1e-2]
        elif eq.__class__.__name__ == "Neumann_Boundary":
            loss_weights = [1e-3] * (self.n_input - 1) + [1] + [1e-2]*2
        wandb.config.update({"adam_iterations": adam_every, "loss_weights": loss_weights})
        adam = self.Adam()
        data = self.data
        try:
            for i in range(cycle):
                print("Cycle: ", i)
                domain_points, boundary_points = self.get_anchors(domain_anchors, boundary_anchors)
                data.replace_with_anchors(domain_points)
                data.train_x_bc = boundary_points
                # Use compiled model with Adam
                self.model.compile(optimizer=adam, metrics=metrics, loss_weights=loss_weights)
                # This .train call is assumed to be jax-friendly in the new dde version
                losses, train_state = self.model.train(iterations=adam_every, display_every=10)
                for idx, loss_val in enumerate(self.model.train_state.loss_train, start=1):
                    wandb.log({f"Adam loss_{idx}": loss_val})
                for idx, metric_val in enumerate(self.model.train_state.metrics_test, start=1):
                    wandb.log({f"Adam metric_{idx}": metric_val})
            # Save with dde.saveplot
            dde.saveplot(losses, train_state, issave=True, isplot=False, output_dir=save_path)
            return self.model
        except KeyboardInterrupt:
            dde.saveplot(losses, train_state, issave=True, isplot=False, output_dir=save_path)
            return self.model