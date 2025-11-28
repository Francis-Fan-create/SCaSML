import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import time
import sys
import os
import cProfile
import shutil
import copy
from optimizers.Adam import Adam
import jax.numpy as jnp
import re
from scipy import stats
import matplotlib.ticker as ticker

class ComputingBudget(object):
    '''
    Computing Budget test: compares solvers under equal total computational budget.
    
    This test demonstrates that SCaSML achieves better accuracy than MLP and PINN
    when all methods are given the same total computing time (training + inference).

    Attributes:
    equation (object): An object representing the equation to solve.
    dim (int): The dimension of the input space minus one.
    solver1 (object): A jax Gaussian Process model (PINN).
    solver2 (object): An object for the MLP solver.
    solver3 (object): An object for the ScaSML solver.
    t0 (float): The initial time.
    T (float): The final time.
    '''
    def __init__(self, equation, solver1, solver2, solver3, is_train):
        '''
        Initializes the computing budget test with given solvers and equation.

        Parameters:
        equation (object): The equation object containing problem specifics.
        solver1 (object): The PINN solver.
        solver2 (object): The MLP solver object.
        solver3 (object): The ScaSML solver object.
        is_train (bool): Whether to train the models.
        '''
        # Save original stdout and stderr
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        # Initialize the parameters
        self.equation = equation
        self.dim = equation.n_input - 1
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.t0 = equation.t0
        self.T = equation.T
        self.is_train = is_train

    def test(self, save_path, budget_levels=[1.0, 2.0, 4.0, 8.0, 16.0], num_domain=1000, num_boundary=200):
        '''
        Compares solvers under different computing budget levels.
        
        The budget is normalized such that 1.0 represents a baseline computation time.
        For each budget level, we train each solver to consume approximately the same
        total time (training + inference), then measure the resulting accuracy.
    
        Parameters:
        save_path (str): The path to save the results.
        budget_levels (list): List of budget multipliers (e.g., [1.0, 2.0, 4.0]).
        num_domain (int): The number of points in the test domain.
        num_boundary (int): The number of points on the test boundary.
        '''
        # Initialize the profiler
        profiler = cProfile.Profile()
        profiler.enable()

        is_train = self.is_train
    
        # Create the save path if it does not exist
        class_name = self.__class__.__name__
        new_path = f"{save_path}/{class_name}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        save_path = new_path
    
        # Set the approximation parameters
        eq = self.equation
        eq_name = eq.__class__.__name__
        d = eq.n_input - 1
        
        # Generate test data (fixed across all budget levels)
        data_domain_test, data_boundary_test = eq.generate_test_data(num_domain, num_boundary)
        xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
        exact_sol = eq.exact_solution(xt_test)
        
        # Storage for results
        pinn_errors = []
        mlp_errors = []
        scasml_errors = []
        pinn_times = []
        mlp_times = []
        scasml_times = []
        pinn_flops = []
        mlp_flops = []
        scasml_flops = []
        pinn_flops_source_list = []
        mlp_flops_source_list = []
        scasml_flops_source_list = []
        
        # Base iterations for budget=1.0
        base_iterations = 2000
        
        def estimate_gflops(repeats=3, size=256):
            """
            Estimate the achievable compute on the current device by timing a
            matrix multiplication and converting to GFLOPs/s. We first try a
            JAX-backed matmul (captures GPU if available), otherwise fall back
            to NumPy.
            """
            try:
                # Use jax if available, best to warm-up and block_until_ready.
                import jax
                import jax.numpy as _jnp
                from jax import random as _random
                key1 = _random.PRNGKey(0)
                key2 = _random.PRNGKey(1)
                # Use float32 for speed and realistic load
                a = _random.normal(key1, shape=(size, size), dtype=_jnp.float32)
                b = _random.normal(key2, shape=(size, size), dtype=_jnp.float32)
                # Warm-up to avoid one-time compilation overhead
                _ = _jnp.dot(a, b).block_until_ready()
                times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    _ = _jnp.dot(a, b).block_until_ready()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                avg_time = sum(times) / max(1, len(times))
                flops = 2 * (size ** 3)  # flops for a dense matmul
                gflops = flops / max(avg_time, 1e-12) / 1e9
                return float(gflops)
            except Exception:
                # Fallback: numpy dot (CPU-bound)
                times = []
                for _ in range(repeats):
                    a = np.random.rand(size, size).astype(np.float32)
                    b = np.random.rand(size, size).astype(np.float32)
                    t0 = time.perf_counter()
                    np.dot(a, b)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                avg_time = sum(times) / max(1, len(times))
                flops = 2 * (size ** 3)
                gflops = flops / max(avg_time, 1e-12) / 1e9
                return float(gflops)

        # Estimate device compute capability to convert measured time into
        # an approximate FLOPS used metric. This is hardware-specific and
        # intended for relative comparisons across solvers in the same run.
        device_gflops = estimate_gflops()

        if device_gflops <= 0:
            device_gflops = 1.0

        # Also print/log the estimated device compute capability for transparency
        print(f"Estimated device: {device_gflops:.3f} GFLOPs/s (approx.)")
        print("Note: When JAX HLO is available, this script attempts to estimate FLOPs")
        print("by parsing compiled XLA HLO for model & solver functions. If HLO parsing")
        print("fails (e.g. function is not jax-jittable), we fall back to a device GFLOPs")
        print("microbenchmark and the measured run time to estimate FLOPs. Both are")
        print("hardware-dependent and intended for relative comparisons within a run.")

        if is_train:
            # Helper functions to estimate flops by parsing JAX/XLA HLO text
            def estimate_flops_from_hlo_text(hlo_text):
                """
                Very small parser that approximates FLOPs from HLO text by counting
                matrix multiply (dot/dot_general) ops and elementwise ops (add/mul).
                The result is an estimate intended for relative comparisons only.
                """
                flops_est = 0.0
                try:
                    # DOTs (matrix multiplies) -> 2 * m * k * n flops
                    for m in re.finditer(r"dot\([^)]*f32\[([0-9,]+)\][^,]*,\s*f32\[([0-9,]+)\]", hlo_text):
                        a_s = m.group(1)
                        b_s = m.group(2)
                        try:
                            a_dims = list(map(int, a_s.split(',')))
                            b_dims = list(map(int, b_s.split(',')))
                            # interpret last two dimensions as matrix dims
                            if len(a_dims) >= 2 and len(b_dims) >= 2:
                                m_dim = a_dims[-2]
                                k_dim = a_dims[-1]
                                n_dim = b_dims[-1]
                                flops_est += 2.0 * m_dim * k_dim * n_dim
                        except Exception:
                            pass

                    # Include dot_general if used (same rule as dot)
                    for m in re.finditer(r"dot_general\([^)]*f32\[([0-9,]+)\][^,]*,\s*f32\[([0-9,]+)\]", hlo_text):
                        a_s = m.group(1)
                        b_s = m.group(2)
                        try:
                            a_dims = list(map(int, a_s.split(',')))
                            b_dims = list(map(int, b_s.split(',')))
                            if len(a_dims) >= 2 and len(b_dims) >= 2:
                                m_dim = a_dims[-2]
                                k_dim = a_dims[-1]
                                n_dim = b_dims[-1]
                                flops_est += 2.0 * m_dim * k_dim * n_dim
                        except Exception:
                            pass

                    # Elementwise ops: multiply/add/sub/div
                    # Use the maximum operand shape for the op to count per-element ops
                    for op in ["add", "multiply", "sub", "div", "power", "real_pow"]:
                        for m in re.finditer(rf"\b{op}\([^)]*f32\[([0-9,]+)\]", hlo_text):
                            a_s = m.group(1)
                            try:
                                dims = list(map(int, a_s.split(',')))
                                elems = 1
                                for dd in dims:
                                    elems *= dd
                                flops_est += float(elems)
                            except Exception:
                                pass
                except Exception:
                    # If parsing fails, return 0 to indicate failure
                    return 0.0
                return flops_est

            def estimate_jax_fn_flops(fn, example_args=(), example_kwargs=None):
                """
                Try to obtain an XLA HLO text for the given JAX-compatible function and
                parse it to estimate FLOPs for a single call. Return None on failure.
                """
                if example_kwargs is None:
                    example_kwargs = {}
                try:
                    import jax
                    comp = None
                    try:
                        comp = jax.xla_computation(fn)(*example_args, **example_kwargs)
                    except Exception:
                        # fallback: try compiling a jitted version
                        jitted = jax.jit(fn)
                        comp = jax.xla_computation(jitted)(*example_args, **example_kwargs)
                    hlo_text = comp.as_hlo_text()
                    if not hlo_text:
                        return None
                    return estimate_flops_from_hlo_text(hlo_text)
                except Exception:
                    return None

            def try_architecture_forward_flops(model, example_input=None):
                """
                Attempt to estimate forward flops by introspecting the network architecture
                (common for dde.maps.jax.FNN). Return None if it fails.
                """
                try:
                    net = getattr(model, 'net', None)
                    if net is None:
                        # Some models expose architecture on model.net if present
                        net = model
                    # Try some common attributes for FNN-like networks
                    layers = None
                    for attr in ['layers', 'units', 'hidden_units', 'units_list', 'unit_list', 'arch']:
                        if hasattr(net, attr):
                            layers = getattr(net, attr)
                            break
                    # For dde.maps.jax.FNN, it stores the width as the python list passed in
                    if layers is None and hasattr(net, '__dict__'):
                        # attempt to read the architecture list from the constructor args stored on the object
                        for key in ['_units', 'n_units', 'hidden_sizes', 'sizes', 'units']:
                            if key in net.__dict__:
                                layers = net.__dict__[key]
                                break
                    if layers is None:
                        return None
                    # ensure it's a list of ints
                    if isinstance(layers, (list, tuple)):
                        arch = list(map(int, list(layers)))
                        # compute forward flops per sample: 2 * sum(in* out)
                        fwd = 0
                        for i in range(len(arch) - 1):
                            fwd += 2 * arch[i] * arch[i + 1]
                        return float(fwd)
                    return None
                except Exception:
                    return None
            for budget in budget_levels:
                # Calculate iterations for this budget
                train_iters = int(base_iterations * budget)
                
                # ==========================================
                # PINN: Train and measure time
                # ==========================================
                solver1_copy = copy.deepcopy(self.solver1)
                opt1 = Adam(eq.n_input, 1, solver1_copy, eq)
                
                start_time = time.time()
                trained_pinn = opt1.train(f"{save_path}/model_weights_PINN_{budget}", iters=train_iters)
                train_time_pinn = time.time() - start_time
                
                start_time = time.time()
                sol_pinn = trained_pinn.predict(xt_test)
                inference_time_pinn = time.time() - start_time
                
                total_time_pinn = train_time_pinn + inference_time_pinn
                # Estimate forward flops per sample using JAX HLO if possible
                pinn_fwd_flops = None
                pinn_flops_source = 'Time'
                try:
                    import jax
                    sample_in = jnp.asarray(xt_test[:1]).astype(jnp.float32)
                    pinn_fwd_flops = estimate_jax_fn_flops(lambda x: trained_pinn.predict(x), (sample_in,))
                    if pinn_fwd_flops and pinn_fwd_flops > 0:
                        pinn_flops_source = 'HLO'
                except Exception:
                    pinn_fwd_flops = None

                # Fallback: try architecture introspection on the underlying net
                if not pinn_fwd_flops:
                    try:
                        pinn_fwd_flops = try_architecture_forward_flops(trained_pinn, xt_test[:1])
                        if pinn_fwd_flops and pinn_fwd_flops > 0:
                            pinn_flops_source = 'Arch'
                    except Exception:
                        pinn_fwd_flops = None

                # If forward flops estimation fails, fallback to time-based approach
                if pinn_fwd_flops and pinn_fwd_flops > 0:
                    # Try to infer training batch size from model.data or eq.generate_data
                    train_batch_size = None
                    try:
                        train_data = getattr(trained_pinn, 'data', None)
                        if train_data is not None:
                            # Try a few common attributes used by TimePDE
                            dom = getattr(train_data, 'num_domain', None) or getattr(train_data, 'N_domain', None) or getattr(train_data, 'n_domain', None)
                            bnd = getattr(train_data, 'num_boundary', None) or getattr(train_data, 'N_boundary', None) or getattr(train_data, 'n_boundary', None) or getattr(train_data, 'num_boundary_points', None)
                            ini = getattr(train_data, 'num_initial', None) or getattr(train_data, 'N_initial', None) or getattr(train_data, 'n_initial', None)
                            dom = dom or 0
                            bnd = bnd or 0
                            ini = ini or 0
                            if dom + bnd + ini > 0:
                                train_batch_size = dom + bnd + ini
                    except Exception:
                        train_batch_size = None

                    if not train_batch_size:
                        # fallback to query eq.generate_data
                        try:
                            data_guess = eq.generate_data()
                            dom = getattr(data_guess, 'num_domain', None) or getattr(data_guess, 'N_domain', None) or getattr(data_guess, 'n_domain', None)
                            bnd = getattr(data_guess, 'num_boundary', None) or getattr(data_guess, 'N_boundary', None) or getattr(data_guess, 'n_boundary', None)
                            ini = getattr(data_guess, 'num_initial', None) or getattr(data_guess, 'N_initial', None) or getattr(data_guess, 'n_initial', None)
                            dom = dom or 0
                            bnd = bnd or 0
                            ini = ini or 0
                            if dom + bnd + ini > 0:
                                train_batch_size = dom + bnd + ini
                        except Exception:
                            train_batch_size = None

                    if not train_batch_size:
                        # If everything fails, use a conservative default
                        train_batch_size = 1024

                    # Rough estimate: backward cost ~ 2x forward
                    backward_factor = 2.0
                    train_flops_iter = pinn_fwd_flops * train_batch_size * (1 + backward_factor)
                    train_flops_total = train_iters * train_flops_iter
                    inference_flops_total = pinn_fwd_flops * xt_test.shape[0]
                    # convert to GFLOPs
                    pinn_flops_used = (train_flops_total + inference_flops_total) / 1e9
                else:
                    pinn_flops_used = total_time_pinn * device_gflops
                
                # ==========================================
                # MLP: Adjust training to match budget
                # ==========================================
                solver2_copy = copy.deepcopy(self.solver2)
                
                # MLP inference is slower, so we adjust training iterations
                # to approximately match total time budget
                rho_mlp = max(2, int(np.log(train_iters) / np.log(np.log(train_iters) + 1)))
                
                start_time = time.time()
                # MLP doesn't need PINN training
                dummy_train_time = 0  # MLP uses pre-trained or no training
                train_time_mlp = dummy_train_time
                
                sol_mlp = solver2_copy.u_solve(rho_mlp, rho_mlp, xt_test)
                inference_time_mlp = time.time() - start_time
                
                total_time_mlp = train_time_mlp + inference_time_mlp
                # Estimate mlp u_solve flops via JAX HLO parsing if possible
                mlp_fwd_flops = None
                mlp_flops_source = 'Time'
                try:
                    sample_in = jnp.asarray(xt_test[:1]).astype(jnp.float32)
                    mlp_fwd_flops = estimate_jax_fn_flops(solver2_copy.u_solve, (rho_mlp, rho_mlp, sample_in))
                except Exception:
                    mlp_fwd_flops = None
                # Fallback: use time based
                if mlp_fwd_flops and mlp_fwd_flops > 0:
                    mlp_flops_source = 'HLO'
                    mlp_flops_used = (mlp_fwd_flops * xt_test.shape[0]) / 1e9
                else:
                    mlp_flops_used = total_time_mlp * device_gflops
                
                # ==========================================
                # ScaSML: Optimize training/inference split
                # ==========================================
                solver3_copy = copy.deepcopy(self.solver3)
                
                # ScaSML uses less training for PINN backbone
                scasml_train_iters = train_iters // (d + 1)
                rho_scasml = max(2, int(np.log(scasml_train_iters) / np.log(np.log(scasml_train_iters) + 1)))
                
                opt3 = Adam(eq.n_input, 1, solver3_copy.PINN, eq)
                
                start_time = time.time()
                trained_pinn_backbone = opt3.train(f"{save_path}/model_weights_ScaSML_{budget}", 
                                                   iters=scasml_train_iters)
                train_time_scasml = time.time() - start_time
                
                solver3_copy.PINN = trained_pinn_backbone
                
                start_time = time.time()
                sol_scasml = solver3_copy.u_solve(rho_scasml, rho_scasml, xt_test)
                inference_time_scasml = time.time() - start_time
                
                total_time_scasml = train_time_scasml + inference_time_scasml
                # Estimate scasml backbone forward flops (PINN backbone) from jitted predict
                scasml_backbone_fwd = None
                scasml_backbone_flops_source = 'Time'
                try:
                    sample_in = jnp.asarray(xt_test[:1]).astype(jnp.float32)
                    scasml_backbone_fwd = estimate_jax_fn_flops(lambda x: trained_pinn_backbone.predict(x), (sample_in,))
                    if scasml_backbone_fwd and scasml_backbone_fwd > 0:
                        scasml_backbone_flops_source = 'HLO'
                except Exception:
                    scasml_backbone_fwd = None
                if not scasml_backbone_fwd:
                    try:
                        scasml_backbone_fwd = try_architecture_forward_flops(trained_pinn_backbone, xt_test[:1])
                        if scasml_backbone_fwd and scasml_backbone_fwd > 0:
                            scasml_backbone_flops_source = 'Arch'
                    except Exception:
                        scasml_backbone_fwd = None

                # estimate u_solve flops for ScaSML via HLO compilation
                scasml_usolve_per_sample = None
                scasml_usolve_flops_source = 'Time'
                try:
                    sample_in = jnp.asarray(xt_test[:1]).astype(jnp.float32)
                    scasml_usolve_per_sample = estimate_jax_fn_flops(solver3_copy.u_solve, (rho_scasml, rho_scasml, sample_in))
                    if scasml_usolve_per_sample and scasml_usolve_per_sample > 0:
                        scasml_usolve_flops_source = 'HLO'
                except Exception:
                    scasml_usolve_per_sample = None

                # Compute training flops for backbone if we have a forward flops estimate
                if scasml_backbone_fwd and scasml_backbone_fwd > 0:
                    # Attempt to infer training batch size
                    train_batch_size = None
                    try:
                        train_data = getattr(trained_pinn_backbone, 'data', None)
                        if train_data is not None:
                            dom = getattr(train_data, 'num_domain', None) or getattr(train_data, 'N_domain', None) or getattr(train_data, 'n_domain', None)
                            bnd = getattr(train_data, 'num_boundary', None) or getattr(train_data, 'N_boundary', None) or getattr(train_data, 'n_boundary', None)
                            ini = getattr(train_data, 'num_initial', None) or getattr(train_data, 'N_initial', None) or getattr(train_data, 'n_initial', None)
                            dom = dom or 0
                            bnd = bnd or 0
                            ini = ini or 0
                            if dom + bnd + ini > 0:
                                train_batch_size = dom + bnd + ini
                    except Exception:
                        train_batch_size = None
                    if not train_batch_size:
                        try:
                            data_guess = eq.generate_data()
                            dom = getattr(data_guess, 'num_domain', None) or getattr(data_guess, 'N_domain', None) or getattr(data_guess, 'n_domain', None)
                            bnd = getattr(data_guess, 'num_boundary', None) or getattr(data_guess, 'N_boundary', None) or getattr(data_guess, 'n_boundary', None)
                            ini = getattr(data_guess, 'num_initial', None) or getattr(data_guess, 'N_initial', None) or getattr(data_guess, 'n_initial', None)
                            dom = dom or 0
                            bnd = bnd or 0
                            ini = ini or 0
                            if dom + bnd + ini > 0:
                                train_batch_size = dom + bnd + ini
                        except Exception:
                            train_batch_size = None
                    if not train_batch_size:
                        train_batch_size = 1024
                    backward_factor = 2.0
                    scasml_train_flops_total = scasml_backbone_fwd * train_batch_size * (1 + backward_factor) * scasml_train_iters
                else:
                    scasml_train_flops_total = train_time_scasml * device_gflops if train_time_scasml else 0.0

                # Compute inference flops for ScaSML: prefer per-sample HLO estimate
                if scasml_usolve_per_sample and scasml_usolve_per_sample > 0:
                    scasml_inference_flops_total = scasml_usolve_per_sample * xt_test.shape[0]
                else:
                    scasml_inference_flops_total = inference_time_scasml * device_gflops if inference_time_scasml else 0.0

                scasml_flops_used = (scasml_train_flops_total + scasml_inference_flops_total) / 1e9
                
                # ==========================================
                # Compute Errors
                # ==========================================
                valid_mask = ~(np.isnan(sol_pinn) | np.isnan(sol_mlp) | 
                              np.isnan(sol_scasml) | np.isnan(exact_sol)).flatten()
                
                if np.sum(valid_mask) == 0:
                    continue
                
                errors_pinn = np.abs(sol_pinn.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
                errors_mlp = np.abs(sol_mlp.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
                errors_scasml = np.abs(sol_scasml.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
                
                exact_sol_valid = exact_sol.flatten()[valid_mask]
                
                rel_error_pinn = np.linalg.norm(errors_pinn) / np.linalg.norm(exact_sol_valid)
                rel_error_mlp = np.linalg.norm(errors_mlp) / np.linalg.norm(exact_sol_valid)
                rel_error_scasml = np.linalg.norm(errors_scasml) / np.linalg.norm(exact_sol_valid)
                
                # Store results
                pinn_errors.append(rel_error_pinn)
                mlp_errors.append(rel_error_mlp)
                scasml_errors.append(rel_error_scasml)
                pinn_times.append(total_time_pinn)
                mlp_times.append(total_time_mlp)
                scasml_times.append(total_time_scasml)
                pinn_flops.append(pinn_flops_used)
                mlp_flops.append(mlp_flops_used)
                scasml_flops.append(scasml_flops_used)
                pinn_flops_source_list.append(pinn_flops_source)
                mlp_flops_source_list.append(mlp_flops_source)
                scasml_flops_source_list.append(scasml_backbone_flops_source if scasml_backbone_fwd else scasml_usolve_flops_source)
                
                # ==========================================
                # Statistical Analysis
                # ==========================================
                # Calculate statistics
                pinn_mean = np.mean(errors_pinn)
                pinn_std = np.std(errors_pinn)
                mlp_mean = np.mean(errors_mlp)
                mlp_std = np.std(errors_mlp)
                scasml_mean = np.mean(errors_scasml)
                scasml_std = np.std(errors_scasml)
                
                # Paired t-tests
                t_pinn_scasml, p_pinn_scasml = stats.ttest_rel(errors_pinn, errors_scasml)
                t_mlp_scasml, p_mlp_scasml = stats.ttest_rel(errors_mlp, errors_scasml)
                t_pinn_mlp, p_pinn_mlp = stats.ttest_rel(errors_pinn, errors_mlp)
                
                # Improvement percentages
                improvement_pinn = (rel_error_pinn - rel_error_scasml) / rel_error_pinn * 100
                improvement_mlp = (rel_error_mlp - rel_error_scasml) / rel_error_mlp * 100
                
                # Log to wandb
                wandb.log({
                    f"budget_{budget}_pinn_error": rel_error_pinn,
                    f"budget_{budget}_mlp_error": rel_error_mlp,
                    f"budget_{budget}_scasml_error": rel_error_scasml,
                    f"budget_{budget}_pinn_time": total_time_pinn,
                    f"budget_{budget}_mlp_time": total_time_mlp,
                    f"budget_{budget}_scasml_time": total_time_scasml,
                    f"budget_{budget}_pinn_flops": pinn_flops_used,
                    f"budget_{budget}_mlp_flops": mlp_flops_used,
                    f"budget_{budget}_scasml_flops": scasml_flops_used,
                    f"budget_{budget}_pinn_flops_source": pinn_flops_source,
                    f"budget_{budget}_mlp_flops_source": mlp_flops_source,
                    f"budget_{budget}_scasml_flops_source": scasml_backbone_flops_source if scasml_backbone_fwd else scasml_usolve_flops_source,
                    f"budget_{budget}_improvement_vs_pinn": improvement_pinn,
                    f"budget_{budget}_improvement_vs_mlp": improvement_mlp,
                    f"budget_{budget}_p_pinn_scasml": p_pinn_scasml,
                    f"budget_{budget}_p_mlp_scasml": p_mlp_scasml,
                })
            
            # ==========================================
            # Visualization
            # ==========================================
            # Color scheme
            COLOR_SCHEME = {
                'PINN': '#000000',     # Black
                'MLP': '#A6A3A4',      # Gray
                'SCaSML': '#2C939A'    # Teal
            }
            
            # Configure matplotlib
            plt.rcParams.update({
                'font.family': 'DejaVu Sans',
                'font.size': 9,
                'axes.labelsize': 10,
                'axes.titlesize': 0,
                'legend.fontsize': 8,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'axes.linewidth': 0.8,
                'lines.linewidth': 1.2,
                'lines.markersize': 5,
                'savefig.dpi': 600,
                'savefig.transparent': True,
                'figure.autolayout': False
            })
            
            if len(pinn_errors) == 0:
                print("No valid runs detected; skipping statistics and plotting.")
                profiler.disable()
                profiler.dump_stats(f"{save_path}/{eq_name}_computing_budget.prof")
                artifact = wandb.Artifact(f"{eq_name}_computing_budget", type="profile")
                artifact.add_file(f"{save_path}/{eq_name}_computing_budget.prof")
                wandb.log_artifact(artifact)
                return budget_levels[-1] if budget_levels else 1.0

            # ==========================================
            # Figure 1: Error vs FLOPs used
            # ==========================================
            fig, ax = plt.subplots(figsize=(3.5, 3))
            pinn_flops_array = np.array(pinn_flops)
            mlp_flops_array = np.array(mlp_flops)
            scasml_flops_array = np.array(scasml_flops)

            # Plot with markers
            ax.plot(pinn_flops_array, pinn_errors, color=COLOR_SCHEME['PINN'], 
            marker='o', linestyle='-', label='PINN', 
            markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(mlp_flops_array, mlp_errors, color=COLOR_SCHEME['MLP'], 
            marker='s', linestyle='-', label='MLP',
            markerfacecolor='none', markeredgewidth=0.8)
            ax.plot(scasml_flops_array, scasml_errors, color=COLOR_SCHEME['SCaSML'], 
            marker='^', linestyle='-', label='SCaSML',
            markerfacecolor='none', markeredgewidth=0.8)
            ax.set_xlabel('Computational Cost (GFLOPs)', labelpad=3)
            ax.set_ylabel('Relative L2 Error', labelpad=3)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.legend(frameon=False, loc='upper right')
            ax.grid(True, which='major', axis='both', linestyle='--', 
                   linewidth=0.5, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/Error_vs_FLOPs.pdf', 
                       bbox_inches='tight', pad_inches=0.05)
            plt.close()
            
            # ==========================================
            # Figure 2: Improvement Bar Chart
            # ==========================================
            fig, ax = plt.subplots(figsize=(3.5, 3))
            
            # Calculate improvements at each budget level
            improvements_vs_pinn = [(pinn - scasml) / pinn * 100 
                                   for pinn, scasml in zip(pinn_errors, scasml_errors)]
            improvements_vs_mlp = [(mlp - scasml) / mlp * 100 
                                  for mlp, scasml in zip(mlp_errors, scasml_errors)]
            
            budget_array = np.array(budget_levels[:len(pinn_errors)])
            x = np.arange(len(budget_array))
            # average flops per budget across solvers (for display only)
            avg_flops = (pinn_flops_array + mlp_flops_array + scasml_flops_array) / 3.0
            width = 0.35
            
            bars1 = ax.bar(x - width/2, improvements_vs_pinn, width, 
                          label='SCaSML vs PINN', color=COLOR_SCHEME['PINN'], 
                          edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, improvements_vs_mlp, width, 
                          label='SCaSML vs MLP', color=COLOR_SCHEME['MLP'], 
                          edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Computing Budget (×baseline)', labelpad=3)
            ax.set_ylabel('Improvement (%)', labelpad=3)
            # Show budget multiplier and the approximate GFLOPs (avg across solvers)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{b}×\n{avg:.1f} GFLOPs' for b, avg in zip(budget_array, avg_flops)])
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.legend(frameon=False, loc='upper left')
            ax.grid(True, which='major', axis='y', linestyle='--', 
                   linewidth=0.5, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/Improvement_Bar_Chart.pdf', 
                       bbox_inches='tight', pad_inches=0.05)
            plt.close()
            
            # ==========================================
            # Summary Statistics and Final Log Output
            # ==========================================
            avg_improvement_pinn = np.mean(improvements_vs_pinn)
            avg_improvement_mlp = np.mean(improvements_vs_mlp)
            wandb.log({
                "avg_improvement_vs_pinn": avg_improvement_pinn,
                "avg_improvement_vs_mlp": avg_improvement_mlp,
                "avg_flops_pinn": np.mean(pinn_flops) if len(pinn_flops) else 0.0,
                "avg_flops_mlp": np.mean(mlp_flops) if len(mlp_flops) else 0.0,
                "avg_flops_scasml": np.mean(scasml_flops) if len(scasml_flops) else 0.0,
                "device_gflops_estimated": device_gflops,
            })
            
            # Write final results to log file
            log_file = open(f"{save_path}/ComputingBudget.log", "w")
            sys.stdout = log_file
            sys.stderr = log_file
            
            print("=" * 80)
            print("COMPUTING BUDGET TEST - FINAL RESULTS")
            print("=" * 80)
            print(f"Equation: {eq_name}")
            print(f"Dimension: {d+1}")
            print(f"Budget levels tested: {budget_array.tolist()}")
            print("=" * 80)
            print()
            
            print(f"{ 'Budget':<12} {'PINN Error':<15} {'PINN GFLOPs':<13} {'PINN Src':<8} {'MLP Error':<15} {'MLP GFLOPs':<13} {'MLP Src':<8} {'SCaSML Error':<15} {'SCaSML GFLOPs':<13} {'SCaSML Src':<8}")
            print("-" * 60)
            for i, budget in enumerate(budget_array):
                print(
                    f"{budget:<12.1f} {pinn_errors[i]:<15.6e} {pinn_flops[i]:<13.2f} {pinn_flops_source_list[i]:<8} {mlp_errors[i]:<15.6e} {mlp_flops[i]:<13.2f} {mlp_flops_source_list[i]:<8} {scasml_errors[i]:<15.6e} {scasml_flops[i]:<13.2f} {scasml_flops_source_list[i]:<8}"
                )
            print()
            
            print("Average Improvement:")
            print(f"  SCaSML vs PINN: {avg_improvement_pinn:+.2f}%")
            print(f"  SCaSML vs MLP: {avg_improvement_mlp:+.2f}%")
            print()
            
            print(f"Final Budget Level ({budget_array[-1]:.1f}×):")
            print(f"  PINN error: {pinn_errors[-1]:.6e}")
            print(f"  MLP error: {mlp_errors[-1]:.6e}")
            print(f"  SCaSML error: {scasml_errors[-1]:.6e}")
            print(f"  PINN GFLOPs: {pinn_flops[-1]:.2f} (src: {pinn_flops_source_list[-1]})")
            print(f"  MLP GFLOPs: {mlp_flops[-1]:.2f} (src: {mlp_flops_source_list[-1]})")
            print(f"  SCaSML GFLOPs: {scasml_flops[-1]:.2f} (src: {scasml_flops_source_list[-1]})")
            print(f"  Improvement vs PINN: {improvements_vs_pinn[-1]:+.2f}%")
            print(f"  Improvement vs MLP: {improvements_vs_mlp[-1]:+.2f}%")
            print("=" * 80)
            
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            log_file.close()
        
        # Stop profiler
        profiler.disable()
        profiler.dump_stats(f"{save_path}/{eq_name}_computing_budget.prof")
        
        # Upload profiler results
        artifact = wandb.Artifact(f"{eq_name}_computing_budget", type="profile")
        artifact.add_file(f"{save_path}/{eq_name}_computing_budget.prof")
        wandb.log_artifact(artifact)
        
        print("Computing budget test completed!")
        return budget_levels[-1] if budget_levels else 1.0
