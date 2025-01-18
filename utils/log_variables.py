import functools
import inspect
import numpy as np
import os

def log_variables(func):
    @functools.wraps(func)
    def wrapper_log_variables(*args, **kwargs):
        # get the stack of the current frame
        stack = inspect.stack()
        eq_name = args[0].equation.__class__.__name__
        # initialize the calling class name
        calling_class_name = ""
        
        # iterate through the stack to find the calling class name
        for frame_info in stack:
            if 'self' in frame_info.frame.f_locals:
                # get the class name of the calling class
                calling_class_name = frame_info.frame.f_locals['self'].__class__.__name__
                if calling_class_name in ['NormalSphere', 'SimpleUniform']:
                    break

        
        # check if the caller is ScaSML or MLP
        if hasattr(args[0], 'net'):
            caller_type = "ScaSML"
        else:
            caller_type = "MLP"
        
        # create the log directory and file path
        log_dir = f'results/{eq_name}/{calling_class_name}/callbacks'
        log_file_path = f'{log_dir}/{caller_type}_{calling_class_name}_function_logs.log'
        
        
        # create the log directory if it does not exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        args_repr = [repr(a) for a in args]                      
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  
        signature = ", ".join(args_repr + kwargs_repr)           
        with open(log_file_path, 'a') as file:
            file.write(f"Calling {func.__name__}({signature})\n")
        
        value = func(*args, **kwargs)
        
        func_locals = inspect.currentframe().f_back.f_locals
        local_vars = {k: v for k, v in func_locals.items() if not k.startswith('__') and k != 'args' and k != 'kwargs'}
        
        with open(log_file_path, 'a') as file:
            file.write(f"{func.__name__} returned {value!r}\n")
            for var, val in local_vars.items():
                if isinstance(val, jnp.ndarray):
                    file.write(f"{var} shape: {val.shape}\n")
                else:
                    file.write(f"{var}: {val}\n")
            file.write("\n")
        return value
    return wrapper_log_variables