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

# Regress out eTIV and sex, then add back the mean to preserve original scale
covariates = [df['Gender_bin'], df['measure_eTIV']]
original_means = df[df.columns[8:10]].mean()
residuals, beta = compute_residuals(df[df.columns[8:10]], covariates)
df[df.columns[8:10]] = residuals + original_means.values

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
out_dir = './results_TIV_sex_regressed'
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

# =========================================================================
# Custom plot: claustrum volume with eTIV overlay on secondary y-axis
# =========================================================================

for clau_col in response_cols:
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Primary axis: claustrum volume by group
    for grp_val, grp_label, color in [(0.0, 'HC', 'steelblue'), (1.0, '22q', 'firebrick')]:
        grp_data = df[df['grouping'] == grp_val]
        ax1.plot(grp_data['age'], grp_data[clau_col], '.', color=color, alpha=0.4, markersize=8, label=grp_label)

        # Spaghetti lines
        for sid in grp_data['subj_id'].unique():
            subj = grp_data[grp_data['subj_id'] == sid].sort_values('age')
            ax1.plot(subj['age'], subj[clau_col], '-', color=color, alpha=0.15, linewidth=0.8)

    ax1.set_xlabel('Age', fontsize=13)
    ax1.set_ylabel('Claustrum Volume (mm³)', fontsize=13, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Secondary axis: eTIV
    ax2 = ax1.twinx()
    for grp_val, grp_label, color, marker in [(0.0, 'HC', 'cornflowerblue', 's'), (1.0, '22q', 'lightcoral', 's')]:
        grp_data = df[df['grouping'] == grp_val]
        ax2.plot(grp_data['age'], grp_data['measure_eTIV'], marker, color=color,
                 alpha=0.15, markersize=4, linestyle='none')

    # eTIV trend lines per group
    for grp_val, grp_label, color in [(0.0, 'HC', 'cornflowerblue'), (1.0, '22q', 'lightcoral')]:
        grp_data = df[df['grouping'] == grp_val]
        z = np.polyfit(grp_data['age'], grp_data['measure_eTIV'], 1)
        p = np.poly1d(z)
        age_range = np.linspace(grp_data['age'].min(), grp_data['age'].max(), 100)
        ax2.plot(age_range, p(age_range), '--', color=color, linewidth=2, label=f'eTIV {grp_label}')

    ax2.set_ylabel('eTIV (mm³)', fontsize=13, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

    title = clau_col.replace('_', ' ').title()
    ax1.set_title(f'{title} with eTIV Overlay', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{clau_col}_with_eTIV.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{clau_col}_with_eTIV.eps'), bbox_inches='tight')
    plt.close(fig)
    print(f"eTIV overlay plot saved: {clau_col}_with_eTIV.png")
