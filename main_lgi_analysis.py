"""
main_lgi_analysis.py

Analyzes the relationship between cortical gyrification (LGI) and
claustrum volume, and tests group differences in LGI trajectories.

1. LGI group trajectories (HC vs 22q over age)
2. Correlations between LGI and claustrum volume
3. Claustrum model with LGI as covariate
"""

import numpy as np
import pandas as pd
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from mixed_models import (
    fit_opt_model,
    fdr_correct,
    plot_models_and_save_results,
    group_calculation_effect,
    inter_calculation_effect,
)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

# =========================================================================
# Load and merge data
# =========================================================================

df = pd.read_csv('all_fs_volumes.csv')
lgi = pd.read_csv('lgi_summary.csv')

df = df.rename(columns={
    'Subject_ID': 'subj_id',
    'Age': 'age',
    'clau_lh_Volume_mm3': 'clau_lh',
    'clau_rh_Volume_mm3': 'clau_rh',
    'Diagnosis_bin': 'grouping'
})

df['clau_lh'] = pd.to_numeric(df['clau_lh'], errors='coerce')
df['clau_rh'] = pd.to_numeric(df['clau_rh'], errors='coerce')

# Parse LGI subject names for merge
def parse_lgi(name):
    parts = name.split('_')
    return int(parts[0]), parts[1]

lgi['subj_id'] = lgi['subject'].apply(lambda x: parse_lgi(x)[0])
lgi['ses'] = lgi['subject'].apply(lambda x: parse_lgi(x)[1])
df['ses'] = df['session'].str.replace('ses-', '')

# Merge
df = df.merge(lgi[['subj_id', 'ses', 'lh_mean_lgi', 'rh_mean_lgi']],
              on=['subj_id', 'ses'], how='left')

# Filter
df = df.dropna(subset=['grouping', 'Gender_bin', 'measure_eTIV', 'age',
                        'clau_lh', 'clau_rh', 'lh_mean_lgi', 'rh_mean_lgi']).reset_index(drop=True)
df = df[~((df['age'] < 25) & (df['clau_lh'] < 500))].reset_index(drop=True)
df = df[df['age'] <= 35].reset_index(drop=True)

print(f"N observations: {len(df)}")
print(f"N subjects: {df['subj_id'].nunique()}")

# =========================================================================
# 1. LGI group trajectories
# =========================================================================

out_dir = './results_lgi'
os.makedirs(out_dir, exist_ok=True)

print("\n" + "=" * 70)
print("1. LGI Group Trajectories")
print("=" * 70)

for hemi, lgi_col in [('LH', 'lh_mean_lgi'), ('RH', 'rh_mean_lgi')]:
    model = smf.mixedlm(f"{lgi_col} ~ age * grouping",
                        data=df, groups=df["subj_id"], re_formula="~age")
    result = model.fit(reml=False)
    print(f"\n{hemi} LGI ~ age * grouping:")
    for term in ['Intercept', 'age', 'grouping', 'age:grouping']:
        print(f"  {term}: coef={result.fe_params[term]:.4f}, p={result.pvalues[term]:.4f}")

# Plot LGI trajectories
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for idx, (hemi, lgi_col) in enumerate([('Left Hemisphere', 'lh_mean_lgi'), ('Right Hemisphere', 'rh_mean_lgi')]):
    ax = axes[idx]
    for grp_val, grp_label, color in [(0.0, 'HC', 'steelblue'), (1.0, '22q', 'firebrick')]:
        grp = df[df['grouping'] == grp_val]
        ax.plot(grp['age'], grp[lgi_col], '.', color=color, alpha=0.4, markersize=8, label=grp_label)
        for sid in grp['subj_id'].unique():
            subj = grp[grp['subj_id'] == sid].sort_values('age')
            ax.plot(subj['age'], subj[lgi_col], '-', color=color, alpha=0.15, linewidth=0.8)
        # Trend line
        z = np.polyfit(grp['age'], grp[lgi_col], 1)
        p = np.poly1d(z)
        age_range = np.linspace(grp['age'].min(), grp['age'].max(), 100)
        ax.plot(age_range, p(age_range), '-', color=color, linewidth=3)
    ax.set_xlabel('Age', fontsize=13)
    ax.set_title(hemi, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
axes[0].set_ylabel('Mean LGI', fontsize=13)
fig.suptitle('Cortical Gyrification (LGI) by Group', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'lgi_trajectories.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("\nLGI trajectory plot saved.")

# =========================================================================
# 2. Correlations between LGI and claustrum volume
# =========================================================================

print("\n" + "=" * 70)
print("2. LGI-Claustrum Correlations")
print("=" * 70)

corr_results = []
for clau_col, clau_label in [('clau_lh', 'L Claustrum'), ('clau_rh', 'R Claustrum')]:
    for lgi_col, lgi_label in [('lh_mean_lgi', 'L LGI'), ('rh_mean_lgi', 'R LGI')]:
        formula = f"{clau_col} ~ {lgi_col} + age + Gender_bin + measure_eTIV"
        model = smf.mixedlm(formula, data=df, groups=df["subj_id"])
        result = model.fit(reml=False)
        z_val = result.tvalues[lgi_col]
        p_val = result.pvalues[lgi_col]
        partial_r = z_val / np.sqrt(z_val**2 + len(df))
        corr_results.append({
            'claustrum': clau_label, 'lgi': lgi_label,
            'coefficient': result.fe_params[lgi_col],
            'partial_r': partial_r, 'z_value': z_val, 'p_value': p_val,
        })
        print(f"  {clau_label} ~ {lgi_label}: r={partial_r:.3f}, p={p_val:.4f}")

corr_df = pd.DataFrame(corr_results)
_, p_corr, _, _ = multipletests(corr_df['p_value'], method='fdr_bh')
corr_df['p_fdr'] = p_corr
corr_df.to_csv(os.path.join(out_dir, 'lgi_claustrum_correlations.csv'), index=False)

print("\nFDR-corrected:")
print(corr_df[['claustrum', 'lgi', 'partial_r', 'p_value', 'p_fdr']].to_string(index=False))

# Plot correlation matrix
r_matrix = np.zeros((2, 2))
p_matrix = np.zeros((2, 2))
for i, clau_label in enumerate(['L Claustrum', 'R Claustrum']):
    for j, lgi_label in enumerate(['L LGI', 'R LGI']):
        row = corr_df[(corr_df['claustrum'] == clau_label) & (corr_df['lgi'] == lgi_label)]
        r_matrix[i, j] = row['partial_r'].values[0]
        p_matrix[i, j] = row['p_fdr'].values[0]

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(r_matrix, cmap='RdBu_r', vmin=-0.2, vmax=0.2, aspect='auto')
ax.set_xticks([0, 1]); ax.set_xticklabels(['L LGI', 'R LGI'], fontsize=12)
ax.set_yticks([0, 1]); ax.set_yticklabels(['L Claustrum', 'R Claustrum'], fontsize=12)
for i in range(2):
    for j in range(2):
        sig = '*' if p_matrix[i, j] < 0.05 else ''
        ax.text(j, i, f'r={r_matrix[i,j]:.3f}{sig}\np={p_matrix[i,j]:.3f}',
                ha='center', va='center', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label='Partial r')
ax.set_title('LGI-Claustrum Correlations\n(* = p_FDR < 0.05)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'lgi_claustrum_correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# =========================================================================
# 3. Claustrum trajectory model with LGI as covariate
# =========================================================================

print("\n" + "=" * 70)
print("3. Claustrum Trajectories Controlling for LGI")
print("=" * 70)

response_cols = ['clau_lh', 'clau_rh']
opts = {
    'orders': [1, 2, 3],
    'm_type': 'slope',
    'alpha': 0.05,
    'response_cols': response_cols,
    'group_col': 'grouping',
    'cov_cols': ['Gender_bin', 'measure_eTIV', 'lh_mean_lgi', 'rh_mean_lgi'],
}

out_model_vect = fit_opt_model(df, opts)

print("\nUncorrected p-values:")
for m in out_model_vect:
    if m is not None:
        gp = m.group_effect['p'] if m.group_effect else None
        ip = m.inter_effect['p'] if m.inter_effect else None
        print(f"  {m.m_name}: group p={gp}, interaction p={ip}")

out_model_vect_corr = fdr_correct(out_model_vect, opts['alpha'])

plot_opts = {
    'leg_txt': ['HC', '22q'],
    'x_label': 'age',
    'y_label': 'Claustrum Volume',
    'plot_ci': True,
    'plot_type': 'redInter',
    'fig_size': (12, 8),
    'n_cov': 4,
}

result_table = plot_models_and_save_results(out_model_vect_corr, plot_opts, 2, out_dir)
print("\nResult table:")
print(result_table)

effect_size_group = group_calculation_effect(out_model_vect)
print("\nGroup Effect Sizes:")
print(effect_size_group)

effect_size_inter = inter_calculation_effect(out_model_vect)
print("\nInteraction Effect Sizes:")
print(effect_size_inter)

print(f"\nAll results saved to {out_dir}/")
