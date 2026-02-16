"""
main_excel_example_glm.py - Port of main_excel_example_glm.m

Example script for fitting GLM trajectories (no random effects)
using data from an Excel file. Suitable for cross-sectional data
without repeated measurements.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; remove this line for interactive use
import matplotlib.pyplot as plt

from mixed_models import (
    fit_opt_model,
    fdr_correct,
    plot_models_and_save_results,
    group_calculation_effect,
    inter_calculation_effect,
)

# Set up matplotlib defaults
plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})


# =========================================================================
# Set up all necessary options here
# =========================================================================

# --- Input data options ---
input_data_file = 'exampleData.xlsx'

col_subj_id = 0
col_age = 1
col_grouping = 2
col_data = [4, 5, 6, 7]
col_cov = [3]

# --- Model estimation options ---
opts = {
    'orders': [0, 1, 2, 3],
    'm_type': 'glm',  # GLM: no random effects
}

# --- Model plotting options ---
out_dir = './results_excel_glm'
save_results = 2

plot_opts = {
    'leg_txt': ['HC', 'Pat'],
    'x_label': 'age',
    'y_label': 'cortical volume',
    'plot_ci': True,
    'plot_type': 'redInter',
}


# =========================================================================
# Execute
# =========================================================================

# --- Read Excel file ---
df = pd.read_excel(input_data_file)
col_names = list(df.columns)
data_in = df.values

input_data = {
    'subj_id': data_in[:, col_subj_id],
    'age': data_in[:, col_age].astype(float),
    'grouping': data_in[:, col_grouping].astype(float),
    'data': data_in[:, col_data].astype(float),
    'cov': data_in[:, col_cov].astype(float) if col_cov else None,
}

if input_data['cov'] is not None:
    input_data['cov'] = input_data['cov'] - np.mean(input_data['cov'], axis=0)

# --- Run model fitting ---
opts['model_names'] = [col_names[c] for c in col_data]
opts['alpha'] = 0.05

out_model_vect = fit_opt_model(input_data, opts)
out_model_vect_corr = fdr_correct(out_model_vect, opts['alpha'])

# --- Plot and save ---
plot_opts['n_cov'] = len(col_cov) if col_cov else 0
plot_models_and_save_results(out_model_vect, plot_opts, save_results, out_dir)

# --- Effect sizes ---
effect_size_group = group_calculation_effect(out_model_vect)
print("\nGroup Effect Sizes:")
print(effect_size_group)

effect_size_inter = inter_calculation_effect(out_model_vect)
print("\nInteraction Effect Sizes:")
print(effect_size_inter)
