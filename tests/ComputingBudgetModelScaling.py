import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import gc
import time
import sys
import os
import cProfile
import shutil
import copy
from optimizers.Adam import Adam
import jax
import inspect
import deepxde as dde
from scipy import stats
from matplotlib.colors import LogNorm


class ComputingBudgetModelScaling(object):
    '''
    Computing Budget test that scales model capacity with computational budget.

    This test aims to be a fairer "equal compute" comparison by allocating
    a given compute budget toward either larger model architectures (for PINN)
    or increased algorithmic complexity (for MLP / ScaSML) instead of simply
    training the same architecture for more epochs.
    '''

    def __init__(self, equation, solver1, solver2, solver3, is_train):
        '''
        Initialize the test object.
        '''
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.equation = equation
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.is_train = is_train
        self.n_input = equation.n_input
        self.n_output = equation.n_output

        # NOTE: This test can be memory-hungry due to large models and training
        # datasets. To avoid GPU OOMs we provide a 'smoke' flag in `test()` and
        # the function will also downscale settings automatically for
        # high-dimensional problems (n_input >= 20). The defaults use conservative
        # settings for typical GPUs but can be tuned by callers via the test
        # parameters.

    def _train_until_converge(self, opt, save_path, max_iters, chunk_iters=200, patience=3, tol=1e-4):
        '''Train the model until convergence or max_iters is reached.

        Returns: trained_model, total_training_time, total_iters
        '''
        total_iters = 0
        no_improve = 0
        last_loss = None
        start_time = time.time()
        while total_iters < max_iters and no_improve < patience:
            # Train in chunks to check convergence
            remaining = max(1, int(min(chunk_iters, max_iters - total_iters)))
            opt.train(save_path, iters=remaining)
            total_iters += remaining
            # Access last training loss
            train_state = opt.model.train_state
            if hasattr(train_state, 'loss_train') and len(train_state.loss_train) > 0:
                loss_now = float(train_state.loss_train[-1])
            else:
                # Can't access loss, break
                break
            if last_loss is None:
                last_loss = loss_now
                no_improve = 0
            else:
                # check relative improvement
                if (last_loss - loss_now) / (abs(last_loss) + 1e-12) > tol:
                    last_loss = loss_now
                    no_improve = 0
                else:
                    no_improve += 1
        total_time = time.time() - start_time
        return opt.model, total_time, total_iters

    def test(self, save_path, budget_levels=None, num_domain=500, num_boundary=100,
             base_iters=1000, base_width=24, base_layers=3, chunk_iters=100, patience=2,
             base_rho=2, base_M=8, max_width_cap=200, batch_predict_size=64,
             smoke=True, param_cap=2_000_000, train_domain=None):
        '''
        Run experiments across budget levels; scale the PINN width and ScaSML backbone
        proportionally to budget while increasing MLP complexity (rho/M) to allocate compute.
        '''
        profiler = cProfile.Profile()
        profiler.enable()

        # Prefer float32 precision to lower memory usage on JAX
        try:
            jax.config.update("jax_enable_x64", False)
        except Exception:
            pass

        eq = self.equation
        eq_name = eq.__class__.__name__

        # Create save path
        class_name = self.__class__.__name__
        new_path = f"{save_path}/{class_name}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        save_path = new_path

        # If budgets weren't provided by the caller, use a conservative default
        if budget_levels is None:
            budget_levels = [1.0, 2.0]

        # If running on very high input dims or in smoke/debug mode, scale down
        # the experiment to avoid excessive GPU memory usage
        if smoke or getattr(self, 'n_input', 0) >= 20:
            num_domain = min(num_domain, 300)
            num_boundary = min(num_boundary, 50)
            base_iters = min(base_iters, 500)
            base_width = min(base_width, 16)
            base_layers = min(base_layers, 3)
            chunk_iters = min(chunk_iters, 50)
            patience = min(patience, 2)
            base_rho = min(base_rho, 1)
            base_M = min(base_M, 2)
            max_width_cap = min(max_width_cap, 200)

        # Determine training set size separately to reduce memory during training
        if train_domain is None:
            # conservative default: keep training size smaller than test size
            train_domain = max(100, min(500, num_domain))

        # Generate test data once
        data_domain_test, data_boundary_test = eq.generate_test_data(num_domain, num_boundary)
        xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
        exact_sol = eq.exact_solution(xt_test)

        pinn_errors = []
        mlp_errors = []
        scasml_errors = []
        pinn_times = []
        mlp_times = []
        scasml_times = []

        # Helpers for batched predictions to avoid storing very large activation
        # graphs on GPU. The batch sizes default to batch_predict_size which can
        # be tuned by callers.
        def _batched_model_predict(model, xt, batch_size):
            n = xt.shape[0]
            out = []
            for i in range(0, n, batch_size):
                arr = np.asarray(model.predict(xt[i:i+batch_size]))
                # Force shape to (batch, n_output)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim == 2:
                    try:
                        arr = arr.reshape(arr.shape[0], self.n_output)
                    except Exception:
                        pass
                out.append(arr)
            return np.vstack(out)

        def _batched_u_solve(u_solve_func, xt, batch_size, pos_args=None, kw_args=None):
            # pos_args: list/tuple of positional args to pass before x_t
            # kw_args: dict of keyword args to pass (excluding x_t)
            n = xt.shape[0]
            out = []
            try:
                sig = inspect.signature(u_solve_func)
            except Exception:
                sig = None
            for i in range(0, n, batch_size):
                x_b = xt[i:i+batch_size]
                try:
                    # prefer using a named x_t parameter if available
                    if sig and ('x_t' in sig.parameters or 'xt' in sig.parameters or 'x' in sig.parameters):
                        call_kwargs = dict()
                        if kw_args:
                            call_kwargs.update(kw_args)
                        # x_t vs xt vs x: prefer x_t
                        if 'x_t' in sig.parameters:
                            call_kwargs['x_t'] = x_b
                            arr = np.asarray(u_solve_func(*(pos_args or []), **call_kwargs))
                            if arr.ndim == 1:
                                arr = arr.reshape(-1, 1)
                            elif arr.ndim == 2:
                                try:
                                    arr = arr.reshape(arr.shape[0], self.n_output)
                                except Exception:
                                    pass
                            out.append(arr)
                        elif 'xt' in sig.parameters:
                            call_kwargs['xt'] = x_b
                            out.append(np.asarray(u_solve_func(*(pos_args or []), **call_kwargs)))
                        elif 'x' in sig.parameters:
                            call_kwargs['x'] = x_b
                            out.append(np.asarray(u_solve_func(*(pos_args or []), **call_kwargs)))
                    else:
                        # Fallback: positional-only signature with data as last positional argument
                        if pos_args:
                            arr = np.asarray(u_solve_func(*pos_args, x_b))
                            if arr.ndim == 1:
                                arr = arr.reshape(-1, 1)
                            elif arr.ndim == 2:
                                try:
                                    arr = arr.reshape(arr.shape[0], self.n_output)
                                except Exception:
                                    pass
                            out.append(arr)
                        else:
                            out.append(np.asarray(u_solve_func(x_b)))
                except TypeError:
                    try:
                        # try to call with only x_b in case other patterns fail
                        arr = np.asarray(u_solve_func(x_b))
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        elif arr.ndim == 2:
                            try:
                                arr = arr.reshape(arr.shape[0], self.n_output)
                            except Exception:
                                pass
                        out.append(arr)
                    except Exception:
                        raise
            return np.vstack(out)

        # iterate budgets
        for budget in budget_levels:
            print(f"\nRunning budget level: {budget}×")
            print(f"Effective settings: train_domain={train_domain}, num_domain={num_domain}, base_iters={base_iters}, base_width={base_width}, base_layers={base_layers}, batch_predict_size={batch_predict_size}")
            # PINN: create a wider model scaled with sqrt(budget)
            width = max(8, int(base_width * (budget ** 0.5)))
            if width > max_width_cap:
                width = max_width_cap
            # Estimate parameters roughly and ensure we don't exceed the param cap
            try:
                approx_params = self.n_input * width + max(0, (base_layers - 1)) * (width ** 2) + width * self.n_output
                if approx_params > param_cap:
                    # Scale width down until approximate params fit under cap
                    new_width = int(max(8, (param_cap // max(base_layers - 1, 1)) ** 0.5))
                    width = min(width, new_width)
                    if width > max_width_cap:
                        width = max_width_cap
            except Exception:
                pass
            net_pinn = dde.maps.jax.FNN([self.n_input] + [width] * base_layers + [self.n_output], "tanh", "Glorot normal")
            # apply terminal transform if available
            if hasattr(eq, 'terminal_transform') and eq.terminal_transform is not None:
                net_pinn.apply_output_transform(eq.terminal_transform)
            # For training, request a smaller dataset to reduce memory
            try:
                data = eq.generate_data(num_domain=train_domain)
            except TypeError:
                # If the equation implementation doesn't accept a num_domain keyword,
                # fall back to the argument-less call (less ideal but supported by some equations)
                data = eq.generate_data()
            model_pinn = dde.Model(data, net_pinn)
            opt_pinn = Adam(self.n_input, self.n_output, model_pinn, eq)
            max_iters = int(base_iters * budget)
            # Train with OOM-aware retry: if we encounter a memory problem, retry
            # with a smaller model size to continue the experiment without failing.
            try:
                trained_pinn, t_pinn, _ = self._train_until_converge(opt_pinn, f"{save_path}/pinn_budget_{budget}",
                                                                     max_iters=max_iters, chunk_iters=chunk_iters, patience=patience)
            except Exception as e:
                msg = str(e).lower()
                if 'out of memory' in msg or 'resourceexhausted' in msg or 'oom' in msg:
                    # Reduce width and optionally reduce training set size, then retry
                    new_width = max(8, int(width // 2))
                    try:
                        new_train_domain = max(50, int(train_domain // 2))
                    except Exception:
                        new_train_domain = max(50, int(num_domain // 2))
                    net_pinn = dde.maps.jax.FNN([self.n_input] + [new_width] * base_layers + [self.n_output], "tanh", "Glorot normal")
                    if hasattr(eq, 'terminal_transform') and eq.terminal_transform is not None:
                        net_pinn.apply_output_transform(eq.terminal_transform)
                    try:
                        new_data = eq.generate_data(num_domain=new_train_domain)
                    except TypeError:
                        new_data = eq.generate_data()
                    model_pinn = dde.Model(new_data, net_pinn)
                    opt_pinn = Adam(self.n_input, self.n_output, model_pinn, eq)
                    trained_pinn, t_pinn, _ = self._train_until_converge(opt_pinn, f"{save_path}/pinn_budget_{budget}_reduced",
                                                                         max_iters=max_iters, chunk_iters=chunk_iters, patience=patience)
                else:
                    raise

            # Evaluate PINN
            start = time.time()
            try:
                sol_pinn = _batched_model_predict(trained_pinn, xt_test, batch_predict_size)
            except Exception as e:
                msg = str(e).lower()
                if 'out of memory' in msg or 'resourceexhausted' in msg or 'oom' in msg:
                    # try again with a smaller batch size
                    try:
                        new_batch = max(8, min(batch_predict_size // 4, 64))
                        sol_pinn = _batched_model_predict(trained_pinn, xt_test, new_batch)
                    except Exception:
                        sol_pinn = np.asarray(trained_pinn.predict(xt_test))
                else:
                    sol_pinn = np.asarray(trained_pinn.predict(xt_test))
            time_pinn_infer = time.time() - start
            # total time p
            total_time_pinn = t_pinn + time_pinn_infer

            # MLP: scale algorithmic complexity by rho and M
            rho_mlp = max(base_rho, int(round(base_rho * budget)))
            M_mlp = max(base_M, int(round(base_M * budget)))
            start = time.time()
            # MLP has two variants (with or without M parameter) -> introspect signature
            try:
                sig = inspect.signature(self.solver2.u_solve)
                # If u_solve accepts named M argument, use a kwargs wrapper
                if 'M' in sig.parameters:
                    try:
                        sol_mlp = _batched_u_solve(self.solver2.u_solve, xt_test, batch_predict_size, pos_args=None, kw_args=dict(n=rho_mlp, rho=rho_mlp, M=M_mlp))
                    except Exception as e:
                        msg = str(e).lower()
                        if 'out of memory' in msg or 'resourceexhausted' in msg or 'oom' in msg:
                            try:
                                sol_mlp = _batched_u_solve(self.solver2.u_solve, xt_test, max(8, batch_predict_size // 8), pos_args=None, kw_args=dict(n=rho_mlp, rho=rho_mlp, M=M_mlp))
                            except Exception:
                                sol_mlp = self.solver2.u_solve(n=rho_mlp, rho=rho_mlp, x_t=xt_test, M=M_mlp)
                        else:
                            sol_mlp = self.solver2.u_solve(n=rho_mlp, rho=rho_mlp, x_t=xt_test, M=M_mlp)
                else:
                    try:
                        sol_mlp = _batched_u_solve(self.solver2.u_solve, xt_test, batch_predict_size, pos_args=[rho_mlp, rho_mlp], kw_args=None)
                    except Exception as e:
                        msg = str(e).lower()
                        if 'out of memory' in msg or 'resourceexhausted' in msg or 'oom' in msg:
                            try:
                                sol_mlp = _batched_u_solve(self.solver2.u_solve, xt_test, max(8, batch_predict_size // 8), pos_args=[rho_mlp, rho_mlp], kw_args=None)
                            except Exception:
                                sol_mlp = self.solver2.u_solve(rho_mlp, rho_mlp, xt_test)
                        else:
                            sol_mlp = self.solver2.u_solve(rho_mlp, rho_mlp, xt_test)
            except (ValueError, TypeError):
                # fallback in case signature introspection fails
                try:
                    sol_mlp = self.solver2.u_solve(n=rho_mlp, rho=rho_mlp, x_t=xt_test, M=M_mlp)
                except TypeError:
                    sol_mlp = self.solver2.u_solve(rho_mlp, rho_mlp, xt_test)
            time_mlp_infer = time.time() - start
            total_time_mlp = time_mlp_infer  # no training for algorithmic solvers

            # ScaSML: scale PINN backbone and train backbone until convergence
            width_scasml = max(8, int(base_width * (budget ** 0.5)))
            if width_scasml > max_width_cap:
                width_scasml = max_width_cap
            net_backbone = dde.maps.jax.FNN([self.n_input] + [width_scasml] * base_layers + [self.n_output], "tanh", "Glorot normal")
            if hasattr(eq, 'terminal_transform') and eq.terminal_transform is not None:
                net_backbone.apply_output_transform(eq.terminal_transform)
            try:
                data_b = eq.generate_data(num_domain=train_domain)
            except TypeError:
                data_b = eq.generate_data()
            model_backbone = dde.Model(data_b, net_backbone)
            opt_backbone = Adam(self.n_input, self.n_output, model_backbone, eq)
            max_iters_backbone = int(base_iters * budget // (eq.n_input))
            try:
                trained_backbone, t_backbone, _ = self._train_until_converge(opt_backbone, f"{save_path}/scasml_backbone_budget_{budget}",
                                                                              max_iters=max_iters_backbone, chunk_iters=chunk_iters, patience=patience)
            except Exception as e:
                msg = str(e).lower()
                if 'out of memory' in msg or 'resourceexhausted' in msg or 'oom' in msg:
                    # Retry with smaller backbone width; keeps training going albeit
                    # with reduced compute.
                    try:
                        new_width_b = max(8, int(width_scasml // 2))
                        # reduce train_domain as a last resort if the retry still OOMs
                        new_train_domain = max(50, int(train_domain // 2))
                        net_backbone = dde.maps.jax.FNN([self.n_input] + [new_width_b] * base_layers + [self.n_output], "tanh", "Glorot normal")
                        if hasattr(eq, 'terminal_transform') and eq.terminal_transform is not None:
                            net_backbone.apply_output_transform(eq.terminal_transform)
                        try:
                            data_b = eq.generate_data(num_domain=new_train_domain)
                        except TypeError:
                            data_b = eq.generate_data()
                        model_backbone = dde.Model(data_b, net_backbone)
                        opt_backbone = Adam(self.n_input, self.n_output, model_backbone, eq)
                        trained_backbone, t_backbone, _ = self._train_until_converge(opt_backbone, f"{save_path}/scasml_backbone_budget_{budget}_reduced",
                                                                                      max_iters=max_iters_backbone, chunk_iters=chunk_iters, patience=patience)
                    except Exception:
                        raise
                else:
                    raise
            # instantiate ScaSML with trained backbone
            scasml_copy = copy.deepcopy(self.solver3)
            scasml_copy.model = trained_backbone
            start = time.time()
            # For scasml, we keep a reasonable rho similar to base_rho and use
            # batched inference to avoid OOMs when xt_test is large
            try:
                sol_scasml = _batched_u_solve(scasml_copy.u_solve, xt_test, batch_predict_size, pos_args=None, kw_args=dict(n=base_rho, rho=base_rho))
            except Exception as e:
                msg = str(e).lower()
                if 'out of memory' in msg or 'resourceexhausted' in msg or 'oom' in msg:
                    try:
                        sol_scasml = _batched_u_solve(scasml_copy.u_solve, xt_test, max(8, batch_predict_size // 8), pos_args=None, kw_args=dict(n=base_rho, rho=base_rho))
                    except Exception:
                        sol_scasml = scasml_copy.u_solve(n=base_rho, rho=base_rho, x_t=xt_test)
                else:
                    sol_scasml = scasml_copy.u_solve(n=base_rho, rho=base_rho, x_t=xt_test)
            time_scasml_infer = time.time() - start
            total_time_scasml = t_backbone + time_scasml_infer

            # Compute errors
            valid_mask = ~(np.isnan(sol_pinn) | np.isnan(sol_mlp) | np.isnan(sol_scasml) | np.isnan(exact_sol)).flatten()
            if np.sum(valid_mask) == 0:
                print("Warning: All predictions are NaN for budget", budget)
                continue
            # Use error arrays
            sol_pinn_valid = sol_pinn.flatten()[valid_mask]
            sol_mlp_valid = sol_mlp.flatten()[valid_mask]
            sol_scasml_valid = sol_scasml.flatten()[valid_mask]
            exact_valid = exact_sol.flatten()[valid_mask]
            errors_pinn = np.abs(sol_pinn_valid - exact_valid)
            errors_mlp = np.abs(sol_mlp_valid - exact_valid)
            errors_scasml = np.abs(sol_scasml_valid - exact_valid)

            rel_error_pinn = np.linalg.norm(errors_pinn) / np.linalg.norm(exact_valid)
            rel_error_mlp = np.linalg.norm(errors_mlp) / np.linalg.norm(exact_valid)
            rel_error_scasml = np.linalg.norm(errors_scasml) / np.linalg.norm(exact_valid)

            pinn_errors.append(rel_error_pinn)
            mlp_errors.append(rel_error_mlp)
            scasml_errors.append(rel_error_scasml)
            pinn_times.append(total_time_pinn)
            mlp_times.append(total_time_mlp)
            scasml_times.append(total_time_scasml)

            # Log per budget
            improvement_pinn = (rel_error_pinn - rel_error_scasml) / rel_error_pinn * 100 if rel_error_pinn != 0 else 0.0
            improvement_mlp = (rel_error_mlp - rel_error_scasml) / rel_error_mlp * 100 if rel_error_mlp != 0 else 0.0
            wandb.log({
                f"budget_{budget}_pinn_error": rel_error_pinn,
                f"budget_{budget}_mlp_error": rel_error_mlp,
                f"budget_{budget}_scasml_error": rel_error_scasml,
                f"budget_{budget}_pinn_time": total_time_pinn,
                f"budget_{budget}_mlp_time": total_time_mlp,
                f"budget_{budget}_scasml_time": total_time_scasml
            })
            # Log improvements this budget
            wandb.log({
                f"budget_{budget}_improvement_vs_pinn": improvement_pinn,
                f"budget_{budget}_improvement_vs_mlp": improvement_mlp
            })
            # Log additional diagnostics for memory tracking and debug
            wandb.log({
                f"budget_{budget}_train_domain": train_domain,
                f"budget_{budget}_pinn_width": width,
                f"budget_{budget}_scasml_backbone_width": width_scasml,
                f"budget_{budget}_batch_predict_size": batch_predict_size
            })

            # Clean up large objects and request GC to release memory for the
            # next budget iteration. This helps avoid OOM on GPUs between runs.
            try:
                del trained_pinn
                del model_pinn
                del net_pinn
                del opt_pinn
            except Exception:
                pass
            try:
                del trained_backbone
                del model_backbone
                del net_backbone
                del opt_backbone
            except Exception:
                pass
            try:
                del sol_pinn, sol_mlp, sol_scasml
            except Exception:
                pass
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Stop profiler
        profiler.disable()
        profiler.dump_stats(f"{save_path}/{eq_name}_computing_budget_model_scaling.prof")
        artifact = wandb.Artifact(f"{eq_name}_computing_budget_model_scaling", type="profile")
        artifact.add_file(f"{save_path}/{eq_name}_computing_budget_model_scaling.prof")
        wandb.log_artifact(artifact)

        # Basic visualizations
        COLOR_SCHEME = {
            'PINN': '#000000',
            'MLP': '#A6A3A4',
            'SCaSML': '#2C939A'
        }
        
        # Plot: errors vs budgets
        budgets_plot = np.array(budget_levels[:len(pinn_errors)])
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(budgets_plot, pinn_errors, marker='o', label='PINN')
        ax.plot(budgets_plot, mlp_errors, marker='s', label='MLP')
        ax.plot(budgets_plot, scasml_errors, marker='^', label='SCaSML')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Computational Budget (×baseline)')
        ax.set_ylabel('Relative L2 Error')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/Error_vs_Budget_ModelScaling.pdf", bbox_inches='tight')
        plt.close()

        # Improvement Bar Chart
        improvements_vs_pinn = [(pinn - scasml) / pinn * 100 if pinn != 0 else 0.0
                               for pinn, scasml in zip(pinn_errors, scasml_errors)]
        improvements_vs_mlp = [(mlp - scasml) / mlp * 100 if mlp != 0 else 0.0
                              for mlp, scasml in zip(mlp_errors, scasml_errors)]

        fig, ax = plt.subplots(figsize=(4, 3))
        x = np.arange(len(budgets_plot))
        width = 0.35
        ax.bar(x - width / 2, improvements_vs_pinn, width, label='SCaSML vs PINN', color=COLOR_SCHEME['PINN'], edgecolor='black')
        ax.bar(x + width / 2, improvements_vs_mlp, width, label='SCaSML vs MLP', color=COLOR_SCHEME['MLP'], edgecolor='black')
        ax.set_xlabel('Computational Budget (×baseline)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b}×' for b in budgets_plot])
        ax.set_ylabel('Improvement (%)')
        ax.axhline(y=0, color='black')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/Improvement_Bar_Chart_ModelScaling.pdf", bbox_inches='tight')
        plt.close()

        # Average improvements
        avg_improvement_pinn = np.mean(improvements_vs_pinn) if len(improvements_vs_pinn) else 0
        avg_improvement_mlp = np.mean(improvements_vs_mlp) if len(improvements_vs_mlp) else 0

        wandb.log({
            "avg_improvement_vs_pinn_model_scaling": avg_improvement_pinn,
            "avg_improvement_vs_mlp_model_scaling": avg_improvement_mlp
        })

        # Return last budget as representative (like other tests)
        return budget_levels[-1]
