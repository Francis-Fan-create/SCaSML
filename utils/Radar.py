import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from math import pi
import matplotlib.colors as mcolors # For generating distinct colors

# ---------------------------
# Data Preparation (Existing Data)
# ---------------------------
systems = ["LCD", "VB-PINN", "VB-GP", "LQG", "DR"]
methods = ["SR", "MLP", "SCaSML"] # Order matters for plotting

l2_errors = {
    "LCD": [0.0524, 0.227, 0.0273],
    "VB-PINN": [0.0117, 0.0836, 0.00397],
    "VB-GP": [0.146, 0.19, 0.0623],
    "LQG": [0.0796, 5.63, 0.0551], # MLP L2 error for LQG
    "DR": [0.00952, 0.0899, 0.00887],
}

linf_errors = {
    "LCD": [0.25, 0.906, 0.161],
    "VB-PINN": [0.0316, 0.296, 0.0216],
    "VB-GP": [0.354, 0.572, 0.254],
    "LQG": [0.778, 12.6, 0.678], # MLP Linf error for LQG (outlier)
    "DR": [0.0751, 0.637, 0.0651],
}

l1_errors = {
    "LCD": [0.0343, 0.167, 0.0178],
    "VB-PINN": [0.00537, 0.0339, 0.00129],
    "VB-GP": [0.0701, 0.08, 0.0248],
    "LQG": [0.14, 12.1, 0.0868], # MLP L1 error for LQG (outlier)
    "DR": [0.0113, 0.0974, 0.0111],
}

# Combine metrics into a single structure - EXCLUDING TIME for radar
data_all_metrics_radar = {}
# Updated metric names for labels
metrics_list_radar = ['Rel L2', 'Linf', 'L1']
# Original keys for data access
metrics_keys_radar = ['L2', 'Linf', 'L1'] # Keep original keys for accessing data dicts

for sys_idx, system in enumerate(systems):
    data_all_metrics_radar[system] = pd.DataFrame({
        # Ensure consistent order matching 'methods' list
        'SR': [l2_errors[system][0], linf_errors[system][0], l1_errors[system][0]],
        'MLP': [l2_errors[system][1], linf_errors[system][1], l1_errors[system][1]],
        'SCaSML': [l2_errors[system][2], linf_errors[system][2], l1_errors[system][2]]
    }, index=metrics_keys_radar) # Use original keys as index

# ---------------------------
# Plotting Settings (Nature Style Refined)
# ---------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans', # Standard sans-serif
    'font.size': 8,           # Base size
    'axes.labelsize': 8,      # Axis labels (metrics)
    'axes.titlesize': 9,      # Plot title size
    'xtick.labelsize': 8,     # Metric labels on radar axis
    'ytick.labelsize': 7,     # Radial tick labels (10^x)
    'legend.fontsize': 8,     # Legend font size (Adjusted for figure legend)
    'axes.linewidth': 0.6,    # Thinner axis lines
    'grid.linewidth': 0.4,    # Thin grid lines
    'grid.color': '#cccccc',  # Lighter grid color
    'grid.alpha': 0.6,        # Slightly more subtle grid
    'lines.linewidth': 1.2,   # Base line width (slightly thicker)
    'lines.markersize': 0,    # No markers
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black', # Keep axis edge for clarity if needed
})

# Updated way to get colormap in newer Matplotlib versions
method_colors = plt.colormaps['Dark2'].colors[:len(methods)]
color_map = {"SR": "#000000", "MLP": "#A6A3A4", "SCaSML": "#2C939A"} # New colors

# Output folder
output_dir = r"utils/Radar"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Figure: Five Radar Charts in a Row (Log Scale, Outlier Handling)
# ---------------------------

num_vars = len(metrics_list_radar)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Create a figure with 1 row and 5 columns of polar subplots
fig_radar, axes = plt.subplots(figsize=(18, 4.5), nrows=1, ncols=len(systems), # Adjusted figsize
                               subplot_kw=dict(polar=True))

# Small epsilon to prevent log10(0)
epsilon = 1e-10

# Define y-axis limits and ticks, setting 10^0 as the upper bound for most lines
min_log_limit = -3.1 # Slightly below -3
max_log_limit_main = 0.0 # Upper limit for most lines (log10(1) = 0)
max_log_limit_plot = 0.1 # Slightly extend plot limit for visual clipping
yticks_log = [-3, -2, -1, 0] # Exponents for 10^-3 to 10^0

outlier_label_part = "MLP" # Just the method part
outlier_system = "LQG"
outlier_style = {'linestyle': '--', 'linewidth': 1.0, 'alpha': 0.8} # Style for outlier

legend_handles = [] # For the figure legend

# Plot each system on its own subplot
for i, system in enumerate(systems):
    ax = axes[i] # Select the current subplot
    data_sys = data_all_metrics_radar[system]

    # Plot lines for each method on the current subplot
    for method_idx, method in enumerate(methods):
        current_color = color_map[method]

        # --- Data Processing: Log10 of errors ---
        errors = data_sys.loc[:, method].values
        log_errors = np.log10(errors + epsilon)

        values = log_errors.flatten().tolist()
        values += values[:1] # Close the loop

        # Handle outlier specifically for MLP-LQG
        is_outlier = (method == outlier_label_part and system == outlier_system)
        if is_outlier:
            # Clip outlier slightly above the main max limit for plotting
            plot_values = np.clip(values, min_log_limit, max_log_limit_plot)
            line, = ax.plot(angles, plot_values, color=current_color,
                            linestyle=outlier_style['linestyle'],
                            linewidth=outlier_style['linewidth'],
                            alpha=outlier_style['alpha'],
                            label=method + ("*" if is_outlier else "")) # Add asterisk only if outlier
        else:
            # Clip other lines strictly at the main max limit
            plot_values = np.clip(values, min_log_limit, max_log_limit_main)
            line, = ax.plot(angles, plot_values, color=current_color,
                            linewidth=plt.rcParams['lines.linewidth'], # Use default linewidth
                            linestyle='solid', label=method)

        # Collect handles for the figure legend (only once)
        if i == 0:
            legend_handles.append(line)

    # --- Formatting the current subplot (ax) ---
    ax.set_ylim(min_log_limit, max_log_limit_plot) # Use plot limit for y-axis range

    # Set metric labels (xticks)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_list_radar, color='black', size=8)

    # Set radial labels (yticks) - Up to 10^0
    yticklabels = [f"$10^{{{int(tick)}}}$" for tick in yticks_log] # Format as powers of 10
    ax.set_yticks(yticks_log) # Set ticks at the integer exponent values
    if i == 0: # Only show y-tick labels on the first plot
        ax.set_yticklabels(yticklabels, color="dimgrey", size=7)
        ax.tick_params(axis='y', pad=-4) # Adjust padding only for the first plot
    else:
        ax.set_yticklabels([]) # Hide y-tick labels on other plots

    # Grid lines
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    # Hide the outer polar spine
    ax.spines['polar'].set_visible(False)

    # Set title for each subplot
    ax.set_title(system, size=9, y=1.15) # Adjust y position

    # Add annotation for the outlier on the specific LQG plot
    if system == outlier_system:
        ax.text(np.pi/2, max_log_limit_plot + 0.6, '*MLP exceeds scale', # Position annotation
                horizontalalignment='center', size=7, color='dimgrey')


# Add a single legend for the entire figure
fig_radar.legend(legend_handles, methods, # Use collected handles and method names
                 loc='lower center', bbox_to_anchor=(0.5, -0.05), # Position below plots
                 ncol=len(methods), # Arrange horizontally
                 frameon=False, fontsize=plt.rcParams['legend.fontsize'])

# Adjust layout to prevent overlap and make space for legend/titles
fig_radar.tight_layout()
fig_radar.subplots_adjust(wspace=0.4, top=0.85, bottom=0.15) # Adjust spacing

# Save the figure
output_filename = "metrics_radar_chart_log_error_row.png"
fig_radar.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
plt.close(fig_radar)

print(f"Combined log error radar chart (5 plots in a row) saved to: {os.path.join(output_dir, output_filename)}")