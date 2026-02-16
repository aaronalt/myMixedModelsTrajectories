"""
main_mat_example.py - Port of main_mat_example.m

Example script for fitting mixed-effect model trajectories
using data from a CSV file (FreeSurfer volumes).

Loads data into a pandas DataFrame, then uses column names throughout.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; remove this line for interactive use
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

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

# --- Load CSV file into a DataFrame ---
df = pd.read_csv('all_fs_volumes.csv')
print(df.head())

# Rename columns to the names expected by fit_opt_model
df = df.rename(columns={
    'Subject_ID': 'subj_id',
    'Age': 'age',
    'clau_lh_Volume_mm3': 'clau_lh',
    'clau_rh_Volume_mm3': 'clau_rh',
})

# Response columns are all volume columns (from column 8 onward in the original CSV)
response_cols = list(df.columns[8:])

# Demean covariates
df['Gender_bin'] = df['Gender_bin'] - df['Gender_bin'].mean()

# --- Model estimation options ---
opts = {
    'orders': [0, 1, 2, 3],
    'm_type': 'slope',
    'alpha': 0.05,
    'response_cols': response_cols,
    'group_col': 'Diagnosis_bin',
    'cov_cols': ['Gender_bin'],  # Assign more covariates here *AARON
}

# --- Model plotting options ---
out_dir = './results_mat_fdrcorr'
save_results = 2

plot_opts = {
    'leg_txt': ['HC', 'Pat'],
    'x_label': 'age',
    'y_label': 'volume',
    'plot_ci': True,
    'plot_type': 'redInter',
    'fig_size': (7.3, 4.3),
    'n_cov': 1,
}


# =========================================================================
# Execute
# =========================================================================

# out_model_vect = fit_opt_model(df, opts)
mixed_lm_model = smf.mixedlm("clau_lh ~ age * group * hemisphere",
                    data=df,
                    groups=df["subj_id"])
out_model_vect = mixed_lm_model.fit()
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
