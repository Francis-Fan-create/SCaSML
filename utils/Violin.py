import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import time
import sys
import os
import cProfile
import shutil
import jax.numpy as jnp
from optimizers.Adam import Adam
from matplotlib.colors import LogNorm
import deepxde as dde
import jax
from solvers import MLP_full_history, ScaSML_full_history
from equations.equations import Grad_Dependent_Nonlinear, Linear_Convection_Diffusion, LQG, Oscillating_Solution

def plot_combined_violin(equations_results, save_path):
    """
    Create a combined violin plot for all equations.
    
    Parameters:
    equations_results (list): List of dictionaries containing results for each equation
                                [{name, errors1, errors2, errors3}, ...]
    save_path (str): Path to save the combined plot
    """
    # =============================================
    # Visualization Configuration
    # =============================================
    COLOR_SCHEME = {
        'PINN': '#000000',     # Black
        'MLP': '#A6A3A4',      # Gray
        'SCaSML': '#2C939A'    # Teal
    }

    # Use DejaVu Sans as a replacement for Arial
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',  # Similar to Arial
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 0,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.linewidth': 0.6,
        'lines.linewidth': 0.8,
        'savefig.dpi': 600,
        'savefig.transparent': True,
        'figure.autolayout': False
    })

    # Create figure with appropriate width for four equation groups
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)

    # Calculate positions for the violin plots
    num_equations = len(equations_results)
    positions = []
    labels = []
    all_errors = []
    
    for i, eq_result in enumerate(equations_results):
        base_pos = i * 4  # 4 positions per equation (3 methods + gap)
        # Positions for PINN, MLP, SCaSML for this equation
        eq_positions = [base_pos + 1, base_pos + 2, base_pos + 3]
        positions.extend(eq_positions)
        
        # Add errors for this equation
        all_errors.extend([eq_result['errors1'], eq_result['errors2'], eq_result['errors3']])
        
        # Labels only at the middle position of each equation group
        labels.extend([''] * 3)
        
    # Create violin plot
    vp = ax.violinplot(all_errors, positions=positions, showmeans=False, showmedians=True)
    
    # Style violins
    colors = []
    for _ in range(num_equations):
        colors.extend(COLOR_SCHEME.values())
    
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
    
    # Configure axes
    ax.set_yscale('log')
    ax.set_ylabel('Absolute Error', labelpad=2)
    
    # Set x-ticks for equation groups
    eq_group_centers = [i * 4 + 2 for i in range(num_equations)]
    eq_labels = [eq['name'] for eq in equations_results]
    ax.set_xticks(eq_group_centers)
    ax.set_xticklabels(eq_labels)
    
    # Add method labels at the bottom
    method_positions = [1, 2, 3]
    for i in range(num_equations):
        offset = i * 4
        for j, method in enumerate(['PINN', 'MLP', 'SCaSML']):
            pos = offset + method_positions[j]
            ax.text(pos, ax.get_ylim()[0] * 1.2, method, ha='center', va='top', 
                    fontsize=6, rotation=45)
    
    # Add vertical separators between equation groups
    for i in range(1, num_equations):
        ax.axvline(x=i*4 - 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Grid and spines
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Final touches
    plt.tight_layout()
    plt.savefig(f"{save_path}/Combined_Error_Distribution.pdf", 
                bbox_inches='tight', pad_inches=0.05)
    plt.close()


if __name__ == "__main__":
    #fix random seed for dde
    dde.config.set_random_seed(1234)
    #use jax backend
    dde.backend.set_default_backend('jax')
    #set default float to float16
    dde.config.set_default_float("float16")
    # fix random seed for jax
    jax.random.PRNGKey(0)
    # device configuration
    device = jax.default_backend()
    print(device)
    if device == 'gpu':
        # get PINNU name
        gpu_name = jax.devices()[0].device_kind

    '''Equation 1: Grad_Dependent_Nonlinear, 20d. Denoted as: VB-PINN, 20d'''
    #initialize the equation
    equation=Grad_Dependent_Nonlinear(n_input=21,n_output=1)
    #initialize the FNN
    #same layer width
    net1=dde.maps.jax.FNN([21]+[50]*5+[1], "tanh", "Glorot normal")
    net2=dde.maps.jax.FNN([21]+[50]*5+[1], "tanh", "Glorot normal")
    net3=dde.maps.jax.FNN([21]+[50]*5+[1], "tanh", "Glorot normal")    
    data1 = equation.generate_data()
    data2 = equation.generate_data()
    data3 = equation.generate_data()
    model1 = dde.Model(data1,net1)
    model2 = dde.Model(data2,net2)
    model3 = dde.Model(data3,net3)
    is_train = True
    #initialize the normal sphere test
    solver1_1= model1 #PINN network
    solver1_2 = model2 #PINN network
    solver1_3 = model3 # PINN network
    solver2=MLP_full_history(equation=equation) #Multilevel Picard object
    solver3_1=ScaSML_full_history(equation=equation,PINN=solver1_1) #ScaSML object
    solver3_2=ScaSML_full_history(equation=equation,PINN=solver1_2) #ScaSML object
    solver3_3=ScaSML_full_history(equation=equation,PINN=solver1_3) #ScaSML object

    '''Equation 2: Linear_Convection_Diffusion, 10d. Denoted as: LCD, 10d'''
    #initialize the equation
    equation=Linear_Convection_Diffusion(n_input=11,n_output=1)
    #initialize the FNN
    #same layer width
    net1=dde.maps.jax.FNN([11]+[50]*5+[1], "tanh", "Glorot normal")
    net2=dde.maps.jax.FNN([11]+[50]*5+[1], "tanh", "Glorot normal")
    net3=dde.maps.jax.FNN([11]+[50]*5+[1], "tanh", "Glorot normal")    
    data1 = equation.generate_data()
    data2 = equation.generate_data()
    data3 = equation.generate_data()
    model1 = dde.Model(data1,net1)
    model2 = dde.Model(data2,net2)
    model3 = dde.Model(data3,net3)
    is_train = True
    #initialize the normal sphere test
    solver1_1= model1 #PINN network
    solver1_2 = model2 #PINN network
    solver1_3 = model3 # PINN network
    solver2=MLP_full_history(equation=equation) #Multilevel Picard object
    solver3_1=ScaSML_full_history(equation=equation,PINN=solver1_1) #ScaSML object
    solver3_2=ScaSML_full_history(equation=equation,PINN=solver1_2) #ScaSML object
    solver3_3=ScaSML_full_history(equation=equation,PINN=solver1_3) #ScaSML object


    '''Equation 3: LQG, 100d. Denoted as: LQG, 100d'''
    #initialize the equation
    equation=LQG(n_input=101,n_output=1)
    #initialize the FNN
    #same layer width
    net1=dde.maps.jax.FNN([101]+[50]*5+[1], "tanh", "Glorot normal")
    net2=dde.maps.jax.FNN([101]+[50]*5+[1], "tanh", "Glorot normal")
    net3=dde.maps.jax.FNN([101]+[50]*5+[1], "tanh", "Glorot normal")    
    data1 = equation.generate_data()
    data2 = equation.generate_data()
    data3 = equation.generate_data()
    model1 = dde.Model(data1,net1)
    model2 = dde.Model(data2,net2)
    model3 = dde.Model(data3,net3)
    is_train = True
    #initialize the normal sphere test
    solver1_1= model1 #PINN network
    solver1_2 = model2 #PINN network
    solver1_3 = model3 # PINN network
    solver2=MLP_full_history(equation=equation) #Multilevel Picard object
    solver3_1=ScaSML_full_history(equation=equation,PINN=solver1_1) #ScaSML object
    solver3_2=ScaSML_full_history(equation=equation,PINN=solver1_2) #ScaSML object
    solver3_3=ScaSML_full_history(equation=equation,PINN=solver1_3) #ScaSML object


    '''Equation 4: Oscillating_Solution, 100d. Denoted as: DR, 100d'''
    #initialize the equation
    equation=Oscillating_Solution(n_input=101,n_output=1)
    #initialize the FNN
    #same layer width
    net1=dde.maps.jax.FNN([101]+[50]*5+[1], "tanh", "Glorot normal")
    net2=dde.maps.jax.FNN([101]+[50]*5+[1], "tanh", "Glorot normal")
    net3=dde.maps.jax.FNN([101]+[50]*5+[1], "tanh", "Glorot normal")    
    data1 = equation.generate_data()
    data2 = equation.generate_data()
    data3 = equation.generate_data()
    model1 = dde.Model(data1,net1)
    model2 = dde.Model(data2,net2)
    model3 = dde.Model(data3,net3)
    is_train = True
    #initialize the normal sphere test
    solver1_1= model1 #PINN network
    solver1_2 = model2 #PINN network
    solver1_3 = model3 # PINN network
    solver2=MLP_full_history(equation=equation) #Multilevel Picard object
    solver3_1=ScaSML_full_history(equation=equation,PINN=solver1_1) #ScaSML object
    solver3_2=ScaSML_full_history(equation=equation,PINN=solver1_2) #ScaSML object
    solver3_3=ScaSML_full_history(equation=equation,PINN=solver1_3) #ScaSML object

    # Make the plot

    results_path = "utils/Violin"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    # Collect results for all equations
    all_equations_results = []
    
    # Equation 1: VB-PINN, 20d
    equation = Grad_Dependent_Nonlinear(n_input=21, n_output=1)
    data_domain_test, data_boundary_test = equation.generate_test_data(1000, 200)
    xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
    exact_sol = equation.exact_solution(xt_test)
    sol1 = solver1_1.predict(xt_test)
    sol2 = solver2.u_solve(2, 2, xt_test)  
    sol3 = solver3_1.u_solve(2, 2, xt_test)  
    valid_mask = ~(np.isnan(sol1) | np.isnan(sol2) | np.isnan(sol3) | np.isnan(exact_sol)).flatten()
    errors1 = np.abs(sol1.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors2 = np.abs(sol2.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors3 = np.abs(sol3.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    all_equations_results.append({"name": "VB-PINN, 20d", "errors1": errors1, "errors2": errors2, "errors3": errors3})
    
    # Equation 2: LCD, 10d
    equation = Linear_Convection_Diffusion(n_input=11, n_output=1)
    data_domain_test, data_boundary_test = equation.generate_test_data(1000, 200)
    xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
    exact_sol = equation.exact_solution(xt_test)
    sol1 = solver1_2.predict(xt_test)
    sol2 = solver2.u_solve(2, 2, xt_test)
    sol3 = solver3_2.u_solve(2, 2, xt_test)
    valid_mask = ~(np.isnan(sol1) | np.isnan(sol2) | np.isnan(sol3) | np.isnan(exact_sol)).flatten()
    errors1 = np.abs(sol1.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors2 = np.abs(sol2.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors3 = np.abs(sol3.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    all_equations_results.append({"name": "LCD, 10d", "errors1": errors1, "errors2": errors2, "errors3": errors3})
    
    # Equation 3: LQG, 100d
    equation = LQG(n_input=101, n_output=1)
    data_domain_test, data_boundary_test = equation.generate_test_data(1000, 200)
    xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
    exact_sol = equation.exact_solution(xt_test)
    sol1 = solver1_3.predict(xt_test)
    sol2 = solver2.u_solve(2, 2, xt_test)
    sol3 = solver3_3.u_solve(2, 2, xt_test)
    valid_mask = ~(np.isnan(sol1) | np.isnan(sol2) | np.isnan(sol3) | np.isnan(exact_sol)).flatten()
    errors1 = np.abs(sol1.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors2 = np.abs(sol2.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors3 = np.abs(sol3.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    all_equations_results.append({"name": "LQG, 100d", "errors1": errors1, "errors2": errors2, "errors3": errors3})
    
    # Equation 4: DR, 100d
    equation = Oscillating_Solution(n_input=101, n_output=1)
    data_domain_test, data_boundary_test = equation.generate_test_data(1000, 200)
    xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
    exact_sol = equation.exact_solution(xt_test)
    sol1 = solver1_3.predict(xt_test)
    sol2 = solver2.u_solve(2, 2, xt_test)
    sol3 = solver3_3.u_solve(2, 2, xt_test)
    valid_mask = ~(np.isnan(sol1) | np.isnan(sol2) | np.isnan(sol3) | np.isnan(exact_sol)).flatten()
    errors1 = np.abs(sol1.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors2 = np.abs(sol2.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors3 = np.abs(sol3.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    all_equations_results.append({"name": "DR, 100d", "errors1": errors1, "errors2": errors2, "errors3": errors3})
    
    # Plot combined violin plot
    plot_combined_violin(all_equations_results, results_path)