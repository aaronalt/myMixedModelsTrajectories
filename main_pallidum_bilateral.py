"""
main_pallidum_bilateral.py

Tests all 4 combinations of claustrum (L/R) x pallidum (L/R) correlations
to check if the relationship is bilateral or lateralized.
"""

import numpy as np
import pandas as pd
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

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

claustrum_cols = ['clau_lh', 'clau_rh']
pallidum_orig = ['subcort_Left-Pallidum', 'subcort_Right-Pallidum']
pallidum_safe = ['pallidum_lh', 'pallidum_rh']
pallidum_labels = ['L Pallidum', 'R Pallidum']

all_numeric = claustrum_cols + pallidum_orig
for col in all_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['pallidum_lh'] = df['subcort_Left-Pallidum']
df['pallidum_rh'] = df['subcort_Right-Pallidum']

df = df.dropna(subset=['grouping', 'Gender_bin', 'measure_eTIV', 'age'] + claustrum_cols + pallidum_safe).reset_index(drop=True)
df = df[~((df['age'] < 25) & (df['clau_lh'] < 500))].reset_index(drop=True)
df = df[df['age'] <= 35].reset_index(drop=True)

print(f"N observations: {len(df)}")
print(f"N subjects: {df['subj_id'].nunique()}")

out_dir = './results_pallidum_bilateral'
os.makedirs(out_dir, exist_ok=True)

# =========================================================================
# All 4 claustrum x pallidum combinations
# =========================================================================

results = []

for clau_col, clau_label in zip(claustrum_cols, ['L Claustrum', 'R Claustrum']):
    for pal_col, pal_label in zip(pallidum_safe, pallidum_labels):
        formula = f"{pal_col} ~ {clau_col} + age + Gender_bin + measure_eTIV"

        try:
            model = smf.mixedlm(formula, data=df, groups=df["subj_id"])
            result = model.fit(reml=False)

            z_val = result.tvalues[clau_col]
            p_val = result.pvalues[clau_col]
            coef = result.fe_params[clau_col]
            n = len(df)
            partial_r = z_val / np.sqrt(z_val**2 + n)

            results.append({
                'claustrum': clau_label,
                'pallidum': pal_label,
                'coefficient': coef,
                'partial_r': partial_r,
                'z_value': z_val,
                'p_value': p_val,
            })
            print(f"  {clau_label} ~ {pal_label}: r={partial_r:.3f}, z={z_val:.2f}, p={p_val:.4f}")
        except Exception as e:
            print(f"  {clau_label} ~ {pal_label}: FAILED ({e})")

results_df = pd.DataFrame(results)

# FDR correction across all 4 tests
valid = ~results_df['p_value'].isna()
if valid.sum() > 0:
    _, p_corr, _, _ = multipletests(results_df.loc[valid, 'p_value'], method='fdr_bh')
    results_df.loc[valid, 'p_fdr'] = p_corr

results_df.to_csv(os.path.join(out_dir, 'pallidum_bilateral_correlations.csv'), index=False)

print("\n" + "="*70)
print("Bilateral Claustrum-Pallidum Correlations (FDR-corrected)")
print("="*70)
print(results_df.to_string(index=False))

# =========================================================================
# Plot: 2x2 matrix heatmap
# =========================================================================

# Build correlation matrix for heatmap
r_matrix = np.zeros((2, 2))
p_matrix = np.zeros((2, 2))
for i, clau_label in enumerate(['L Claustrum', 'R Claustrum']):
    for j, pal_label in enumerate(['L Pallidum', 'R Pallidum']):
        row = results_df[(results_df['claustrum'] == clau_label) & (results_df['pallidum'] == pal_label)]
        if len(row) > 0:
            r_matrix[i, j] = row['partial_r'].values[0]
            p_matrix[i, j] = row['p_fdr'].values[0]

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(r_matrix, cmap='RdBu_r', vmin=-0.15, vmax=0.15, aspect='auto')

ax.set_xticks([0, 1])
ax.set_xticklabels(['L Pallidum', 'R Pallidum'], fontsize=12)
ax.set_yticks([0, 1])
ax.set_yticklabels(['L Claustrum', 'R Claustrum'], fontsize=12)

# Annotate cells with r values and significance
for i in range(2):
    for j in range(2):
        sig = '*' if p_matrix[i, j] < 0.05 else ''
        ax.text(j, i, f'r={r_matrix[i,j]:.3f}{sig}\np={p_matrix[i,j]:.3f}',
                ha='center', va='center', fontsize=11, fontweight='bold')

plt.colorbar(im, ax=ax, label='Partial r')
ax.set_title('Claustrum-Pallidum Bilateral Correlations\n(* = p_FDR < 0.05)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'pallidum_bilateral_heatmap.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'pallidum_bilateral_heatmap.eps'), bbox_inches='tight')
print(f"\nFigures saved to {out_dir}/")
