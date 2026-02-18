"""
main_pallidum_covariate.py

Fits claustrum trajectory models with pallidum volume added as
a covariate, to test whether group effects persist after controlling
for pallidum.
"""

import numpy as np
import pandas as pd
import matplotlib
import os

matplotlib.use('Agg')
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
# Load and prepare data
# =========================================================================

df = pd.read_csv('all_fs_volumes.csv')

df = df.rename(columns={
    'Subject_ID': 'subj_id',
    'Age': 'age',
    'clau_lh_Volume_mm3': 'clau_lh',
    'clau_rh_Volume_mm3': 'clau_rh',
    'Diagnosis_bin': 'grouping'
})

response_cols = list(df.columns[8:10])
df[df.columns[8:10]] = df[df.columns[8:10]].apply(pd.to_numeric, errors='coerce')

# Also convert pallidum columns to numeric
df['subcort_Left-Pallidum'] = pd.to_numeric(df['subcort_Left-Pallidum'], errors='coerce')
df['subcort_Right-Pallidum'] = pd.to_numeric(df['subcort_Right-Pallidum'], errors='coerce')

# Rename pallidum columns (hyphens cause issues)
df['pallidum_lh'] = df['subcort_Left-Pallidum']
df['pallidum_rh'] = df['subcort_Right-Pallidum']

# --- Model estimation options (pallidum added as covariates) ---
opts = {
    'orders': [1, 2, 3],
    'm_type': 'slope',
    'alpha': 0.05,
    'response_cols': response_cols,
    'group_col': 'grouping',
    'cov_cols': ['Gender_bin', 'measure_eTIV', 'pallidum_lh', 'pallidum_rh'],
}

# --- Model plotting options ---
out_dir = './results_pallidum_covariate'
os.makedirs(out_dir, exist_ok=True)
save_results = 2

plot_opts = {
    'leg_txt': ['HC', '22q'],
    'x_label': 'age',
    'y_label': 'Claustrum Volume',
    'plot_ci': True,
    'plot_type': 'redInter',
    'fig_size': (12, 8),
    'n_cov': 4,
}

# =========================================================================
# Execute
# =========================================================================

df = df.dropna(subset=['grouping', 'Gender_bin', 'measure_eTIV', 'age', 'pallidum_lh', 'pallidum_rh']).reset_index(drop=True)
df = df[~((df['age'] < 25) & (df['clau_lh'] < 500))].reset_index(drop=True)
df = df[df['age'] <= 35].reset_index(drop=True)

print(f"N observations: {len(df)}")
print(f"N subjects: {df['subj_id'].nunique()}")

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
