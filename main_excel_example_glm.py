"""
main_excel_example_glm.py - Port of main_excel_example_glm.m

Example script for fitting GLM trajectories (no random effects)
using data from an Excel file. Suitable for cross-sectional data
without repeated measurements.

The input is a pandas DataFrame — columns are referenced by name, not index.
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

# --- Read Excel file into a DataFrame ---
input_data_file = 'exampleData.xlsx'
df = pd.read_excel(input_data_file)

# Rename required columns
df = df.rename(columns={
    df.columns[0]: 'subj_id',
    df.columns[1]: 'age',
})

group_col = df.columns[2]
cov_cols = [df.columns[3]]
response_cols = list(df.columns[4:8])

# Demean covariates
for c in cov_cols:
    df[c] = df[c] - df[c].mean()

# --- Model estimation options ---
opts = {
    'orders': [0, 1, 2, 3],
    'm_type': 'glm',              # GLM: no random effects
    'alpha': 0.05,
    'response_cols': response_cols,
    'group_col': group_col,
    'cov_cols': cov_cols,
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
    'n_cov': len(cov_cols),
}


# =========================================================================
# Execute
# =========================================================================

out_model_vect = fit_opt_model(df, opts)
out_model_vect_corr = fdr_correct(out_model_vect, opts['alpha'])

result_table = plot_models_and_save_results(out_model_vect_corr, plot_opts, save_results, out_dir)
print("\nResult table:")
print(result_table)

effect_size_group = group_calculation_effect(out_model_vect)
print("\nGroup Effect Sizes:")
print(effect_size_group)

effect_size_inter = inter_calculation_effect(out_model_vect)
print("\nInteraction Effect Sizes:")
print(effect_size_inter)
