"""
main_excel_example.py - Port of main_excel_example.m

Example script for fitting mixed-effect model trajectories
using data from an Excel file.

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

# --- Read Excel file into a DataFrame ---
input_data_file = 'exampleData.xlsx'
df = pd.read_excel(input_data_file)

print(f"Loaded {len(df)} rows from {input_data_file}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# --- Map your column names ---
# Rename columns to the names expected by fit_opt_model:
#   'subj_id' and 'age' are required.
#   Grouping and covariate columns keep their original names.
df = df.rename(columns={
    df.columns[0]: 'subj_id',   # first column = subject IDs
    df.columns[1]: 'age',       # second column = age
})

# Column names for grouping, covariates, and response variables
group_col = df.columns[2]                # e.g. 'diagnosis' (0/1 for 2 groups)
                                          # Set to None if you have only 1 group
                                          # Pass a list of 2 columns for 3 groups
cov_cols = [df.columns[3]]               # e.g. ['sex']
response_cols = list(df.columns[4:8])    # e.g. ['vol_1', 'vol_2', 'vol_3', 'vol_4']

# Demean covariates in-place
for c in cov_cols:
    df[c] = df[c] - df[c].mean()

# --- Model estimation options ---
opts = {
    'orders': [0, 1, 2, 3],      # 0=constant, 1=linear, 2=quadratic, 3=cubic
    'm_type': 'slope',            # 'intercept', 'slope' (recommended), or 'glm'
    'alpha': 0.05,
    'response_cols': response_cols,
    'group_col': group_col,
    'cov_cols': cov_cols,
    'model_names': response_cols,  # display names (default = column names)
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
    'n_cov': len(cov_cols),
}


# =========================================================================
# Execute the model estimation and plot/save results
# =========================================================================

# Fit models — pass the DataFrame directly
out_model_vect = fit_opt_model(df, opts)

# Correct for multiple comparisons using FDR
out_model_vect_corr = fdr_correct(out_model_vect, opts['alpha'])

# Plot and save models
result_table = plot_models_and_save_results(out_model_vect_corr, plot_opts, save_results, out_dir)
print("\nResult table:")
print(result_table)

# Calculate effect sizes
effect_size_group = group_calculation_effect(out_model_vect)
print("\nGroup Effect Sizes:")
print(effect_size_group)

effect_size_inter = inter_calculation_effect(out_model_vect)
print("\nInteraction Effect Sizes:")
print(effect_size_inter)
