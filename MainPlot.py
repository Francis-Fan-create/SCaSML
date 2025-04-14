import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import os

# ---------------------------
# Data Preparation
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

comp_time = {
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

df_l2 = pd.DataFrame(l2_errors, index=methods).T
df_time = pd.DataFrame(comp_time, index=methods).T
df_linf = pd.DataFrame(linf_errors, index=methods).T
df_l1 = pd.DataFrame(l1_errors, index=methods).T

# Heatmap dataframe
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

# ---------------------------
# Plotting Settings
# ---------------------------
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

palette_default = {"SR": "#888888", "MLP": "#66a3d2", "SCaSML": "#1f4e79"}
custom_palette = {"SR": "#000000", "MLP": "#A6A3A4", "SCaSML": "#2C939A"}

# Output folder
output_dir = "exported_figures"
os.makedirs(output_dir, exist_ok=True)

# # ---------------------------
# # Default composite figure
# # ---------------------------
# fig1 = plt.figure(figsize=(12, 14))
# gs1 = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.2])

# # Panel A
# ax0 = plt.subplot(gs1[0])
# df_l2.plot(kind='bar', logy=True, ax=ax0, color=[palette_default[m] for m in df_l2.columns])
# ax0.set_ylabel("Relative L² Error (log scale)")
# ax0.set_title("A. Relative L² Error Across PDE Systems")
# ax0.legend(title="Method")
# ax0.set_xlabel("")
# ax0.grid(True, axis='y', linestyle='--', alpha=0.6)

# # Panel B
# ax1 = plt.subplot(gs1[1])
# df_time.plot(kind='bar', logy=True, ax=ax1, color=[palette_default[m] for m in df_time.columns])
# ax1.set_ylabel("Computation Time (s, log scale)")
# ax1.set_title("B. Total Computation Time")
# ax1.legend(title="Method")
# ax1.set_xlabel("")
# ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

# # Panel C
# ax2 = plt.subplot(gs1[2])
# sns.heatmap(df_heatmap, annot=True, fmt=".2e", cmap="Blues", cbar_kws={"label": "Error Value"}, ax=ax2)
# ax2.set_title("C. Error Metrics Across Methods and Systems")
# ax2.set_ylabel("Method-System")
# ax2.set_xlabel("Metric")

# plt.tight_layout()
# plt.subplots_adjust(hspace=0.3)
# fig1.savefig(os.path.join(output_dir, "composite_figure_v1.png"), dpi=300)


# ---------------------------
# Figure 1: L2 Error and Computation Time
# ---------------------------
fig1 = plt.figure(figsize=(10, 6))

# Combined Plot: L2 + Time
ax_comb = plt.gca()
x = np.arange(len(df_l2.index))
bar_width = 0.25

# Bar for L2
for i, method in enumerate(df_l2.columns):
    ax_comb.bar(x + i * bar_width - bar_width, df_l2[method], 
                width=bar_width, label=f"{method} - L² Error", 
                color=custom_palette[method], alpha=0.8)

ax_comb.set_ylabel("Relative L² Error (log scale)")
ax_comb.set_yscale("log")
ax_comb.set_xticks(x)
ax_comb.set_xticklabels(df_l2.index)
ax_comb.set_title("Relative L² Error and Computation Time")

# Twin axis for Time
ax_comb_time = ax_comb.twinx()
for i, method in enumerate(df_time.columns):
    ax_comb_time.plot(x + i * bar_width - bar_width, df_time[method],
                      marker='o', linestyle='--', label=f"{method} - Time",
                      color=custom_palette[method])
ax_comb_time.set_ylabel("Computation Time (s, log scale)")
ax_comb_time.set_yscale("log")

# Combined Legend
handles1, labels1 = ax_comb.get_legend_handles_labels()
handles2, labels2 = ax_comb_time.get_legend_handles_labels()
ax_comb.legend(handles1 + handles2, labels1 + labels2, loc="upper center", 
               bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25, right=0.88)  
fig1.savefig(os.path.join(output_dir, "l2_error_time_figure.png"), dpi=300)

# ---------------------------
# Figure 2: Error Metrics Heatmap
# ---------------------------
fig2 = plt.figure(figsize=(10, 8))
ax_heatmap = plt.gca()

sns.heatmap(df_heatmap, annot=True, fmt=".1e", cmap="Blues_r", norm=LogNorm(),
            cbar_kws={"label": "Error (log scale)"}, ax=ax_heatmap)
ax_heatmap.set_title("Error Metrics Across Methods and Systems")
ax_heatmap.set_ylabel("Method-System")
ax_heatmap.set_xlabel("Metric")

plt.tight_layout()
fig2.savefig(os.path.join(output_dir, "error_metrics_heatmap.png"), dpi=300)

print("Figures saved to:", os.path.abspath(output_dir))