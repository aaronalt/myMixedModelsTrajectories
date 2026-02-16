"""
main_excel_example.py - Port of main_excel_example.m

Example script for fitting mixed-effect model trajectories
using data from an Excel file.
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

# Set up matplotlib defaults for nicer plots
plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})


# =========================================================================
# Set up all necessary options here (input file, output directory, etc.)
# =========================================================================

# --- Input data options ---
input_data_file = 'exampleData.xlsx'

col_subj_id = 0    # column index in Excel file with subject IDs
col_age = 1         # column index with age
col_grouping = 2    # column index for grouping (0/1 values)
                    # Set to None if you have only 1 group
                    # 1 column for 2 groups
                    # 2 columns for 3 groups
col_data = [4, 5, 6, 7]  # column indices with your data
col_cov = [3]       # column indices with covariates (can be empty list)

# --- Model estimation options ---
opts = {
    'orders': [0, 1, 2, 3],  # model orders: 0=constant, 1=linear, 2=quadratic, 3=cubic
    'm_type': 'slope',        # 'intercept', 'slope' (recommended), or 'glm'
}

# --- Model plotting options ---
out_dir = './results_excel'
save_results = 2  # 0=No, 1=table only, 2=table + plots

plot_opts = {
    'leg_txt': ['HC', 'Pat'],         # group legend labels
    'x_label': 'age',
    'y_label': 'cortical volume',
    'plot_ci': True,                   # plot confidence intervals
    'plot_type': 'redInter',           # 'full', 'redInter', or 'redGrp'
    # 'plot_col': [np.array([0.4, 0.76, 0.65]),  # custom colors (optional)
    #              np.array([0.99, 0.55, 0.38]),
    #              np.array([0.55, 0.63, 0.80])],
}


# =========================================================================
# Execute the model estimation and plot/save results
# =========================================================================

# --- Read Excel file ---
df = pd.read_excel(input_data_file)
print(f"Loaded {len(df)} rows from {input_data_file}")
print(f"Columns: {list(df.columns)}")

# --- Prepare input data ---
data_in = df.values  # numeric data
col_names = list(df.columns)

input_data = {
    'subj_id': data_in[:, col_subj_id],
    'age': data_in[:, col_age].astype(float),
    'grouping': data_in[:, col_grouping].astype(float) if col_grouping is not None else None,
    'data': data_in[:, col_data].astype(float),
    'cov': data_in[:, col_cov].astype(float) if col_cov else None,
}

# Demean covariates
if input_data['cov'] is not None:
    input_data['cov'] = input_data['cov'] - np.mean(input_data['cov'], axis=0)

# --- Run model fitting ---
opts['model_names'] = [col_names[c] for c in col_data]
opts['alpha'] = 0.05

# Fit models
out_model_vect = fit_opt_model(input_data, opts)

# Correct for multiple comparisons using FDR
out_model_vect_corr = fdr_correct(out_model_vect, opts['alpha'])

# --- Plot and save models ---
plot_opts['n_cov'] = len(col_cov) if col_cov else 0
plot_models_and_save_results(out_model_vect, plot_opts, save_results, out_dir)

# --- Calculate effect sizes ---
effect_size_group = group_calculation_effect(out_model_vect)
print("\nGroup Effect Sizes:")
print(effect_size_group)

effect_size_inter = inter_calculation_effect(out_model_vect)
print("\nInteraction Effect Sizes:")
print(effect_size_inter)
