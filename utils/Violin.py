import sys
import os
# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import os
import deepxde as dde
import jax
import wandb
from solvers.MLP_full_history import MLP_full_history
from solvers.ScaSML_full_history import ScaSML_full_history
from optimizers.Adam import Adam
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
    
    # Get current y limits for positioning labels
    # y_min = ax.get_ylim()[0] # No longer needed for method label positioning
    
    # Set x-ticks for equation groups with adjusted padding
    eq_group_centers = [i * 4 + 2 for i in range(num_equations)]
    eq_labels = [eq['name'] for eq in equations_results]
    ax.set_xticks(eq_group_centers)
    ax.set_xticklabels(eq_labels)
    
    # Remove the ylim adjustment based on y_min
    # ax.set_ylim(bottom=y_min * 3.0) 
    
    # Add method labels below the x-axis, without rotation
    method_positions = [1, 2, 3]
    # Define a vertical offset in axis coordinates (e.g., 15% below the axis)
    y_offset = -0.02
    for i in range(num_equations):
        offset = i * 4
        for j, method in enumerate(['PINN', 'MLP', 'SCaSML']):
            pos = offset + method_positions[j]
            # Use axis coordinates for y positioning (ax.transAxes)
            # Place text below the axis (y < 0), centered horizontally, no rotation
            ax.text(pos, y_offset, method, 
                    transform=ax.get_xaxis_transform(), # Use x data coords, y axis coords
                    ha='center', va='top', 
                    fontsize=6) # Removed rotation, adjusted font size if needed
    
    # ... [existing code for vertical separators, grid, spines] ...
    
    # Adjust equation labels position (x-axis tick labels)
    ax.tick_params(axis='x', pad=5) # Adjust padding as needed (maybe less now)
    
    # Final touches - adjust bottom margin
    plt.tight_layout(pad=1.5) # Adjust overall padding
    plt.subplots_adjust(bottom=0.2) # Ensure enough space for labels below axis
    plt.savefig(f"{save_path}/Combined_Error_Distribution.pdf", 
                bbox_inches='tight', pad_inches=0.1,dpi=1000) # Adjust savefig padding
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
    #initialize wandb
    wandb.init(project="Violin", notes="All", tags=["Adam training","L_inf_training"],mode="disabled") #debug mode
    wandb.config.update({"device": device}) # record device type

    '''Equation 1: Grad_Dependent_Nonlinear, 80d. Denoted as: VB-PINN, 80d'''
    #initialize the equation
    equation1=Grad_Dependent_Nonlinear(n_input=81,n_output=1)
    #initialize the FNN
    #same layer width
    net1=dde.maps.jax.FNN([81]+[50]*5+[1], "tanh", "Glorot normal") 
    data1 = equation1.generate_data()
    model1 = dde.Model(data1,net1)
    is_train = True
    #initialize the normal sphere test
    solver1_1= model1 #PINN network
    solver2_1=MLP_full_history(equation=equation1) #Multilevel Picard object
    solver3_1=ScaSML_full_history(equation=equation1,PINN=solver1_1) #ScaSML object

    '''Equation 2: Linear_Convection_Diffusion, 60d. Denoted as: LCD, 60d'''
    #initialize the equation
    equation2=Linear_Convection_Diffusion(n_input=61,n_output=1)
    #initialize the FNN
    #same layer width
    net2=dde.maps.jax.FNN([61]+[50]*5+[1], "tanh", "Glorot normal") 
    data2 = equation2.generate_data()
    model2 = dde.Model(data2,net2)
    is_train = True
    #initialize the normal sphere test
    solver1_2 = model2 #PINN network
    solver2_2=MLP_full_history(equation=equation2) #Multilevel Picard object
    solver3_2=ScaSML_full_history(equation=equation2,PINN=solver1_2) #ScaSML object


    '''Equation 3: LQG, 160d. Denoted as: LQG, 160d'''
    #initialize the equation
    equation3=LQG(n_input=161,n_output=1)
    #initialize the FNN
    #same layer width
    net3=dde.maps.jax.FNN([161]+[50]*5+[1], "tanh", "Glorot normal")    
    data3 = equation3.generate_data()
    model3 = dde.Model(data3,net3)
    is_train = True
    #initialize the normal sphere test
    solver1_3= model3 #PINN network
    solver2_3=MLP_full_history(equation=equation3) #Multilevel Picard object
    solver3_3=ScaSML_full_history(equation=equation3,PINN=solver1_3) #ScaSML object


    '''Equation 4: Oscillating_Solution, 160d. Denoted as: DR, 160d'''
    #initialize the equation
    equation4=Oscillating_Solution(n_input=161,n_output=1)
    #initialize the FNN
    #same layer width
    net4=dde.maps.jax.FNN([161]+[50]*5+[1], "tanh", "Glorot normal") 
    data4 = equation4.generate_data()
    model4 = dde.Model(data4,net4)
    is_train = True
    #initialize the normal sphere test
    solver1_4= model4 #PINN network
    solver2_4=MLP_full_history(equation=equation4) #Multilevel Picard object
    solver3_4=ScaSML_full_history(equation=equation4,PINN=solver1_4) #ScaSML object

    # Make the plot

    results_path = "utils/Violin"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    # Collect results for all equations
    all_equations_results = []
    
    # Equation 1: VB-PINN, 80d
    if is_train:
        opt = Adam(equation1.n_input, 1, solver1_1, equation1)
        trained_model1= opt.train(f"{results_path}/model_weights_Adam")
        solver1_1 = trained_model1
        solver3_1.PINN = trained_model1     
    data_domain_test, data_boundary_test = equation1.generate_test_data(1000, 200)
    xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
    exact_sol = equation1.exact_solution(xt_test)
    sol1 = solver1_1.predict(xt_test)
    sol2 = solver2_1.u_solve(2, 2, xt_test)  
    sol3 = solver3_1.u_solve(2, 2, xt_test)  
    valid_mask = ~(np.isnan(sol1) | np.isnan(sol2) | np.isnan(sol3) | np.isnan(exact_sol)).flatten()
    errors1 = np.abs(sol1.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors2 = np.abs(sol2.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors3 = np.abs(sol3.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    all_equations_results.append({"name": "VB-PINN, 80d", "errors1": errors1, "errors2": errors2, "errors3": errors3})
    
    # Equation 2: LCD, 60d
    if is_train:
        opt = Adam(equation2.n_input, 1, solver1_2, equation2)
        trained_model2= opt.train(f"{results_path}/model_weights_Adam")
        solver1_2 = trained_model2
        solver3_2.PINN = trained_model2  
    data_domain_test, data_boundary_test = equation2.generate_test_data(1000, 200)
    xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
    exact_sol = equation2.exact_solution(xt_test)
    sol1 = solver1_2.predict(xt_test)
    sol2 = solver2_2.u_solve(2, 2, xt_test)
    sol3 = solver3_2.u_solve(2, 2, xt_test)
    valid_mask = ~(np.isnan(sol1) | np.isnan(sol2) | np.isnan(sol3) | np.isnan(exact_sol)).flatten()
    errors1 = np.abs(sol1.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors2 = np.abs(sol2.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors3 = np.abs(sol3.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    all_equations_results.append({"name": "LCD, 60d", "errors1": errors1, "errors2": errors2, "errors3": errors3})
    
    # Equation 3: LQG, 160d
    if is_train:
        opt = Adam(equation1.n_input, 1, solver1_3, equation3)
        trained_model3= opt.train(f"{results_path}/model_weights_Adam")
        solver1_3 = trained_model3
        solver3_3.PINN = trained_model3  
    data_domain_test, data_boundary_test = equation3.generate_test_data(1000, 200)
    xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
    exact_sol = equation3.exact_solution(xt_test)
    sol1 = solver1_3.predict(xt_test)
    sol2 = solver2_3.u_solve(2, 2, xt_test)
    sol3 = solver3_3.u_solve(2, 2, xt_test)
    valid_mask = ~(np.isnan(sol1) | np.isnan(sol2) | np.isnan(sol3) | np.isnan(exact_sol)).flatten()
    errors1 = np.abs(sol1.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors2 = np.abs(sol2.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors3 = np.abs(sol3.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    all_equations_results.append({"name": "LQG, 160d", "errors1": errors1, "errors2": errors2, "errors3": errors3})
    
    # Equation 4: DR, 160d
    if is_train:
        opt = Adam(equation1.n_input, 1, solver1_4, equation4)
        trained_model4= opt.train(f"{results_path}/model_weights_Adam")
        solver1_4 = trained_model4
        solver3_4.PINN = trained_model4  
    data_domain_test, data_boundary_test = equation4.generate_test_data(1000, 200)
    xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
    exact_sol = equation4.exact_solution(xt_test)
    sol1 = solver1_4.predict(xt_test)
    sol2 = solver2_4.u_solve(2, 2, xt_test)
    sol3 = solver3_4.u_solve(2, 2, xt_test)
    valid_mask = ~(np.isnan(sol1) | np.isnan(sol2) | np.isnan(sol3) | np.isnan(exact_sol)).flatten()
    errors1 = np.abs(sol1.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors2 = np.abs(sol2.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    errors3 = np.abs(sol3.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
    all_equations_results.append({"name": "DR, 160d", "errors1": errors1, "errors2": errors2, "errors3": errors3})
    
    # Plot combined violin plot
    plot_combined_violin(all_equations_results, results_path)