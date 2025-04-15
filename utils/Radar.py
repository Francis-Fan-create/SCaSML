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
    'legend.fontsize': 7,     # Legend font size
    'axes.linewidth': 0.6,    # Thinner axis lines
    'grid.linewidth': 0.4,    # Thin grid lines
    'grid.color': '#cccccc',  # Lighter grid color
    'grid.alpha': 0.6,        # Slightly more subtle grid
    'lines.linewidth': 1.0,   # Base line width
    'lines.markersize': 0,    # No markers
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black', # Keep axis edge for clarity if needed
})

# Generate a larger, distinct color palette for 15 lines
num_lines = len(systems) * len(methods)
colors = plt.cm.get_cmap('tab20').colors[:num_lines] # Get first 15 colors from tab20


# Output folder
output_dir = r"utils/Radar"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Figure: Single Radar Chart (Log Scale, Outlier Handling)
# ---------------------------

num_vars = len(metrics_list_radar)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Create a single figure and a single polar subplot
fig_radar, ax = plt.subplots(figsize=(8, 8),
                             subplot_kw=dict(polar=True))

# Small epsilon to prevent log10(0)
epsilon = 1e-10

# Define y-axis limits and ticks, setting 10^0 as the upper bound for most lines
min_log_limit = -3.1 # Slightly below -3
max_log_limit_main = 0.0 # Upper limit for most lines (log10(1) = 0)
max_log_limit_plot = 0.1 # Slightly extend plot limit for visual clipping
yticks_log = [-3, -2, -1, 0] # Exponents for 10^-3 to 10^0

outlier_label = "MLP-LQG"
outlier_style = {'linestyle': '--', 'linewidth': 0.8, 'alpha': 0.7} # Style for outlier

color_index = 0
legend_handles = []
legend_labels = []

# Plot all lines on the single axis
for system in systems:
    data_sys = data_all_metrics_radar[system]
    for method in methods:
        current_label = f"{method}-{system}"
        current_color = colors[color_index]

        # --- Data Processing: Log10 of errors ---
        errors = data_sys.loc[:, method].values
        log_errors = np.log10(errors + epsilon)

        values = log_errors.flatten().tolist()
        values += values[:1] # Close the loop

        # Handle outlier
        if current_label == outlier_label:
            # Clip outlier slightly above the main max limit for plotting
            plot_values = np.clip(values, min_log_limit, max_log_limit_plot)
            line, = ax.plot(angles, plot_values, color=current_color,
                            linestyle=outlier_style['linestyle'],
                            linewidth=outlier_style['linewidth'],
                            alpha=outlier_style['alpha'],
                            label=current_label + "*") # Add asterisk to label
            legend_handles.append(line)
            legend_labels.append(current_label + "*")
        else:
            # Clip other lines strictly at the main max limit
            plot_values = np.clip(values, min_log_limit, max_log_limit_main)
            line, = ax.plot(angles, plot_values, color=current_color,
                            linewidth=plt.rcParams['lines.linewidth'], # Use default linewidth
                            linestyle='solid', label=current_label)
            legend_handles.append(line)
            legend_labels.append(current_label)

        color_index += 1

# --- Formatting the single axis ---
ax.set_ylim(min_log_limit, max_log_limit_plot) # Use plot limit for y-axis range

# Set metric labels (xticks)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_list_radar, color='black', size=8)

# Set radial labels (yticks) - Up to 10^0
yticklabels = [f"$10^{{{int(tick)}}}$" for tick in yticks_log] # Format as powers of 10
ax.set_yticks(yticks_log) # Set ticks at the integer exponent values
ax.set_yticklabels(yticklabels, color="dimgrey", size=7)
ax.tick_params(axis='y', pad=-4)

# Grid lines
ax.xaxis.grid(True)
ax.yaxis.grid(True)

# Hide the outer polar spine
ax.spines['polar'].set_visible(False)

# Set title
ax.set_title("Comparison Across All Systems and Methods (Log Error Scale)", size=9, y=1.12)

# Add annotation for the outlier
ax.text(np.pi/2, max_log_limit_plot + 0.5, '*MLP-LQG exceeds scale', # Position annotation
        horizontalalignment='center', size=7, color='dimgrey')

# Add a legend outside the plot area
fig_radar.legend(legend_handles, legend_labels, # Use collected handles/labels
                 loc='center left', bbox_to_anchor=(0.8, 0.5), ncol=1,
                 frameon=False, fontsize=10)

# Adjust layout
fig_radar.tight_layout()
fig_radar.subplots_adjust(right=0.75, top=0.9) # Adjust right for legend, top for annotation

# Save the figure
fig_radar.savefig(os.path.join(output_dir, "metrics_radar_chart_log_error_outlier_handled.png"), dpi=300, bbox_inches='tight')
plt.close(fig_radar)

print(f"Combined log error radar chart (outlier handled) saved to: {os.path.join(output_dir, 'metrics_radar_chart_log_error_outlier_handled.png')}")