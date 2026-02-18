"""
main_correlations_by_group.py

Partial correlations between claustrum and pallidum volumes,
run separately for HC and 22q groups.
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

roi_cols = [
    'subcort_Left-Pallidum', 'subcort_Right-Pallidum',
]
claustrum_cols = ['clau_lh', 'clau_rh']
all_numeric_cols = claustrum_cols + roi_cols

for col in all_numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['grouping', 'Gender_bin', 'measure_eTIV', 'age'] + all_numeric_cols).reset_index(drop=True)
df = df[~((df['age'] < 25) & (df['clau_lh'] < 500))].reset_index(drop=True)
df = df[df['age'] <= 35].reset_index(drop=True)

# Sanitize column names
df['subcort_Left_Pallidum'] = df['subcort_Left-Pallidum']
df['subcort_Right_Pallidum'] = df['subcort_Right-Pallidum']
safe_rois = ['subcort_Left_Pallidum', 'subcort_Right_Pallidum']
roi_labels = ['L Pallidum', 'R Pallidum']

out_dir = './results_correlations_by_group'
os.makedirs(out_dir, exist_ok=True)

# =========================================================================
# Run correlations separately by group
# =========================================================================

groups = {0.0: 'HC', 1.0: '22q'}
all_results = []

for grp_val, grp_label in groups.items():
    df_grp = df[df['grouping'] == grp_val].reset_index(drop=True)
    print(f"\n{'='*50}")
    print(f"Group: {grp_label} (N={len(df_grp)}, subjects={df_grp['subj_id'].nunique()})")
    print('='*50)

    for clau_col in claustrum_cols:
        for safe_roi, roi_label in zip(safe_rois, roi_labels):
            formula = f"{safe_roi} ~ {clau_col} + age + Gender_bin + measure_eTIV"

            try:
                model = smf.mixedlm(formula, data=df_grp, groups=df_grp["subj_id"])
                result = model.fit(reml=False)

                z_val = result.tvalues[clau_col]
                p_val = result.pvalues[clau_col]
                n = len(df_grp)
                partial_r = z_val / np.sqrt(z_val**2 + n)

                all_results.append({
                    'group': grp_label,
                    'claustrum': clau_col,
                    'roi': roi_label,
                    'partial_r': partial_r,
                    'z_value': z_val,
                    'p_value': p_val,
                    'n_obs': n,
                })
                print(f"  {clau_col} ~ {roi_label}: r={partial_r:.3f}, p={p_val:.4f}")
            except Exception as e:
                print(f"  {clau_col} ~ {roi_label}: FAILED ({e})")

results_df = pd.DataFrame(all_results)

# FDR correction within each group
for grp_label in groups.values():
    mask = results_df['group'] == grp_label
    valid = mask & ~results_df['p_value'].isna()
    if valid.sum() > 0:
        _, p_corr, _, _ = multipletests(results_df.loc[valid, 'p_value'], method='fdr_bh')
        results_df.loc[valid, 'p_fdr'] = p_corr

results_df.to_csv(os.path.join(out_dir, 'pallidum_correlations_by_group.csv'), index=False)

print("\n" + "="*70)
print("Results by Group (FDR-corrected within group)")
print("="*70)
print(results_df[['group', 'claustrum', 'roi', 'partial_r', 'p_value', 'p_fdr', 'n_obs']].to_string(index=False))

# =========================================================================
# Plot: grouped bar chart
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, clau_col in enumerate(claustrum_cols):
    ax = axes[idx]
    subset = results_df[results_df['claustrum'] == clau_col]

    x = np.arange(len(roi_labels))
    width = 0.35

    hc_data = subset[subset['group'] == 'HC']
    q22_data = subset[subset['group'] == '22q']

    bars1 = ax.bar(x - width/2, hc_data['partial_r'].values, width, label='HC', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, q22_data['partial_r'].values, width, label='22q', color='firebrick', edgecolor='black')

    # Significance markers
    for i, (r, p) in enumerate(zip(hc_data['partial_r'].values, hc_data['p_fdr'].values)):
        if p < 0.05:
            ax.text(x[i] - width/2, r + 0.005 * np.sign(r), '*', ha='center', fontsize=14, fontweight='bold')
    for i, (r, p) in enumerate(zip(q22_data['partial_r'].values, q22_data['p_fdr'].values)):
        if p < 0.05:
            ax.text(x[i] + width/2, r + 0.005 * np.sign(r), '*', ha='center', fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(roi_labels)
    ax.set_ylabel('Partial r')
    ax.set_title(clau_col.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)

fig.suptitle('Claustrum-Pallidum Correlations by Group\n(* = p_FDR < 0.05)', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'pallidum_correlations_by_group.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'pallidum_correlations_by_group.eps'), bbox_inches='tight')
print(f"\nFigures saved to {out_dir}/")
