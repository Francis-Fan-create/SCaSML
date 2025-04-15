import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import os
from math import pi

# ---------------------------
# Data Preparation (Existing Data)
# ---------------------------
systems = ["LCD", "VB-PINN", "VB-GP", "LQG", "DR"]
methods = ["SR", "MLP", "SCaSML"]

l2_errors = {
    "LCD": [0.0524, 0.227, 0.0273],
    "VB-PINN": [0.0117, 0.0836, 0.00397],
    "VB-GP": [0.146, 0.19, 0.0623],
    "LQG": [0.0796, 5.63, 0.0551],
    "DR": [0.00952, 0.0899, 0.00887],
}

comp_time = { # Keep for other figures if needed, but won't be used in radar
    "LCD": [2.64, 11.24, 23.75],
    "VB-PINN": [1.15, 7.05, 13.82],
    "VB-GP": [1.67, 11.08, 63.38],
    "LQG": [1.54, 8.67, 26.95],
    "DR": [1.62, 7.75, 60.86],
}

linf_errors = {
    "LCD": [0.25, 0.906, 0.161],
    "VB-PINN": [0.0316, 0.296, 0.0216],
    "VB-GP": [0.354, 0.572, 0.254],
    "LQG": [0.778, 12.6, 0.678],
    "DR": [0.0751, 0.637, 0.0651],
}

l1_errors = {
    "LCD": [0.0343, 0.167, 0.0178],
    "VB-PINN": [0.00537, 0.0339, 0.00129],
    "VB-GP": [0.0701, 0.08, 0.0248],
    "LQG": [0.14, 12.1, 0.0868],
    "DR": [0.0113, 0.0974, 0.0111],
}

# Combine metrics into a single structure - EXCLUDING TIME for radar
data_all_metrics_radar = {}
# Use more descriptive metric names for labels if desired
metrics_list_radar = ['L2 Error', 'Linf Error', 'L1 Error']
# Original keys for data access
metrics_keys_radar = ['L2', 'Linf', 'L1']

for sys_idx, system in enumerate(systems):
    data_all_metrics_radar[system] = pd.DataFrame({
        'SR': [l2_errors[system][0], linf_errors[system][0], l1_errors[system][0]],
        'MLP': [l2_errors[system][1], linf_errors[system][1], l1_errors[system][1]],
        'SCaSML': [l2_errors[system][2], linf_errors[system][2], l1_errors[system][2]]
    }, index=metrics_keys_radar) # Use original keys as index

# ---------------------------
# Plotting Settings (Nature Style Refined)
# ---------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,           # Base font size
    'axes.labelsize': 8,      # Axis labels (like metric names on radar)
    'axes.titlesize': 9,      # Subplot titles
    'xtick.labelsize': 7,     # X-tick labels (metric names here)
    'ytick.labelsize': 6,     # Y-tick labels (radial axis)
    'legend.fontsize': 7,     # Legend font size
    'axes.linewidth': 0.7,    # Thinner axis lines
    'grid.linewidth': 0.4,    # Thinner grid lines
    'lines.linewidth': 1.2,   # Slightly thinner plot lines
    'lines.markersize': 0,    # No markers on radar lines
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'grid.color': '#cccccc',  # Lighter grey grid
    'grid.alpha': 0.6,
    'axes.spines.left':   True, # Ensure spines are visible if needed elsewhere
    'axes.spines.bottom': True,
    'axes.spines.top':    False, # Hide top/right spines generally
    'axes.spines.right':  False,
})

custom_palette = {"SR": "#000000", "MLP": "#A6A3A4", "SCaSML": "#2C939A"}

# Output folder
output_dir = r"utils/BarAndHeat"
os.makedirs(output_dir, exist_ok=True)

# --- Previous Figures Code (Keep if needed) ---
# Figure 1: L2 Error and Computation Time
fig1 = plt.figure(figsize=(10, 6))
ax_comb = plt.gca()
df_l2 = pd.DataFrame(l2_errors, index=methods).T
df_time = pd.DataFrame(comp_time, index=methods).T # Still need df_time here
x = np.arange(len(df_l2.index))
bar_width = 0.25
for i, method in enumerate(df_l2.columns):
    ax_comb.bar(x + i * bar_width - bar_width, df_l2[method],
                width=bar_width, label=f"{method} - L² Error",
                color=custom_palette[method], alpha=0.8)
ax_comb.set_ylabel("Relative L² Error (log scale)")
ax_comb.set_yscale("log")
ax_comb.set_xticks(x)
ax_comb.set_xticklabels(df_l2.index)
ax_comb.set_title("Relative L² Error and Computation Time")
ax_comb_time = ax_comb.twinx()
for i, method in enumerate(df_time.columns):
    ax_comb_time.plot(x + i * bar_width - bar_width, df_time[method],
                      marker='o', linestyle='--', label=f"{method} - Time",
                      color=custom_palette[method])
ax_comb_time.set_ylabel("Computation Time (s, log scale)")
ax_comb_time.set_yscale("log")
handles1, labels1 = ax_comb.get_legend_handles_labels()
handles2, labels2 = ax_comb_time.get_legend_handles_labels()
first_legend = ax_comb.legend(handles1, labels1, loc="upper center",
                             bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True)
ax_comb.add_artist(first_legend)
second_legend = ax_comb.legend(handles2, labels2, loc="upper center",
                              bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=True)
plt.tight_layout()
plt.subplots_adjust(bottom=0.3, right=0.88)
fig1.savefig(os.path.join(output_dir, "l2_error_time_figure.png"), dpi=300)
plt.close(fig1)

# Figure 2: Error Metrics Heatmap
fig2 = plt.figure(figsize=(10, 8))
ax_heatmap = plt.gca()
df_linf = pd.DataFrame(linf_errors, index=methods).T # Still need df_linf here
df_l1 = pd.DataFrame(l1_errors, index=methods).T # Still need df_l1 here
heatmap_data = []
for system in systems:
    for method in methods:
        heatmap_data.append([
            f"{method}-{system}",
            l2_errors[system][methods.index(method)],
            linf_errors[system][methods.index(method)],
            l1_errors[system][methods.index(method)],
        ])
df_heatmap = pd.DataFrame(heatmap_data, columns=["Method-System", "L2", "Linf", "L1"])
df_heatmap.set_index("Method-System", inplace=True)
sns.heatmap(df_heatmap, annot=True, fmt=".1e", cmap="Blues_r", norm=LogNorm(),
            cbar_kws={"label": "Error (log scale)"}, ax=ax_heatmap)
ax_heatmap.set_title("Error Metrics Across Methods and Systems")
ax_heatmap.set_ylabel("Method-System")
ax_heatmap.set_xlabel("Metric")
plt.tight_layout()
fig2.savefig(os.path.join(output_dir, "error_metrics_heatmap.png"), dpi=300)
plt.close(fig2)

print("Previous figures saved.")