import functools
import inspect
import numpy as np
import os
from datetime import datetime

def log_variables(func):
    @functools.wraps(func)
    def wrapper_log_variables(*args, **kwargs):
        # get the class name of the first argument
        self_arg = args[0]
        class_name = self_arg.equation.__class__.__name__  
        
        # detect the solvers name
        if hasattr(self_arg, 'net'):
            caller_type="ScaML"
        else:
            caller_type="MLP"
        # get the log file path
        log_dir = os.path.join('results', f"{class_name}_callbacks")
        log_file_path = os.path.join(log_dir, f'{caller_type}_function_logs.log')
        
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
                if isinstance(val, np.ndarray):
                    file.write(f"{var} shape: {val.shape}\n")
                else:
                    file.write(f"{var}: {val}\n")
            file.write("\n")
        return value
    return wrapper_log_variables