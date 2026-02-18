"""
main_mat_example_residualized.py

Same as main_covaried.py but regresses out eTIV and sex from
claustrum volumes using compute_residuals before fitting the model.
Adds back the mean so volumes stay in original scale.
"""

import numpy as np
import pandas as pd
import matplotlib
import os

from functions.compute_residuals import compute_residuals

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
    'Diagnosis_bin': 'grouping'
})

# Response columns are all volume columns (from column 8 onward in the original CSV)
response_cols = list(df.columns[8:10])
df[df.columns[8:10]] = df[df.columns[8:10]].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values before residualizing
df = df.dropna(subset=['grouping', 'Gender_bin', 'measure_eTIV', 'age']).reset_index(drop=True)
df = df.dropna(subset=list(df.columns[8:10])).reset_index(drop=True)

# Regress out eTIV, sex, and contralateral hemisphere from each claustrum
# Save raw values before modifying
raw_lh = df['clau_lh'].values.copy()
raw_rh = df['clau_rh'].values.copy()

# LH: regress out eTIV, sex, and RH claustrum
cov_lh = [df['Gender_bin'].values, df['measure_eTIV'].values, raw_rh]
resid_lh, _ = compute_residuals(raw_lh.reshape(-1, 1), cov_lh)
df['clau_lh'] = resid_lh.ravel() + raw_lh.mean()

# RH: regress out eTIV, sex, and LH claustrum
cov_rh = [df['Gender_bin'].values, df['measure_eTIV'].values, raw_lh]
resid_rh, _ = compute_residuals(raw_rh.reshape(-1, 1), cov_rh)
df['clau_rh'] = resid_rh.ravel() + raw_rh.mean()

# --- Model estimation options (no covariates since they've been regressed out) ---
opts = {
    'orders': [1, 2, 3],
    'm_type': 'slope',
    'alpha': 0.05,
    'response_cols': response_cols,
    'group_col': 'grouping',
    'cov_cols': [],
}

# --- Model plotting options ---
out_dir = './results_TIV_sex_hemi_regressed'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
save_results = 2

plot_opts = {
    'leg_txt': ['HC', '22q'],
    'x_label': 'age',
    'y_label': 'Claustrum Volume',
    'plot_ci': True,
    'plot_type': 'redInter',
    'fig_size': (12, 8),
    'n_cov': 0,
}


# =========================================================================
# Execute
# =========================================================================

df = df[~((df['age'] < 25) & (df['clau_lh'] < 500))].reset_index(drop=True)
df = df[df['age'] <= 35].reset_index(drop=True)
out_model_vect = fit_opt_model(df, opts)

# Print uncorrected p-values
print("\nUncorrected p-values:")
for m in out_model_vect:
    if m is not None:
        gp = m.group_effect['p'] if m.group_effect else None
        ip = m.inter_effect['p'] if m.inter_effect else None
        print(f"  {m.m_name}: group p={gp}, interaction p={ip}")

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
