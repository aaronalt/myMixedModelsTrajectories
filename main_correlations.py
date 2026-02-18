"""
main_correlations.py

Partial correlations between claustrum volumes and basal ganglia / cingulate
cortex regions, controlling for age, sex, and eTIV.

Uses a mixed-effects approach to account for repeated measures (longitudinal data).
"""

import numpy as np
import pandas as pd
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

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

# Regions of interest
basal_ganglia = [
    'subcort_Left-Accumbens-area', 'subcort_Right-Accumbens-area',
    'subcort_Left-Caudate', 'subcort_Right-Caudate',
    'subcort_Left-Pallidum', 'subcort_Right-Pallidum',
    'subcort_Left-Putamen', 'subcort_Right-Putamen',
]

cingulate = [
    'cort_lh_caudalanteriorcingulate', 'cort_rh_caudalanteriorcingulate',
    'cort_lh_isthmuscingulate', 'cort_rh_isthmuscingulate',
    'cort_lh_posteriorcingulate', 'cort_rh_posteriorcingulate',
    'cort_lh_rostralanteriorcingulate', 'cort_rh_rostralanteriorcingulate',
]

roi_cols = basal_ganglia + cingulate
claustrum_cols = ['clau_lh', 'clau_rh']

# Convert to numeric
all_numeric_cols = claustrum_cols + roi_cols
for col in all_numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing values
df = df.dropna(subset=['grouping', 'Gender_bin', 'measure_eTIV', 'age'] + all_numeric_cols).reset_index(drop=True)

# Remove outlier and restrict age (same as main analysis)
df = df[~((df['age'] < 25) & (df['clau_lh'] < 500))].reset_index(drop=True)
df = df[df['age'] <= 35].reset_index(drop=True)

print(f"N observations: {len(df)}")
print(f"N subjects: {df['subj_id'].nunique()}")

# =========================================================================
# Partial correlations using mixed-effects models
# For each ROI, fit: ROI ~ claustrum + age + sex + eTIV + (1|subj_id)
# The t-statistic on claustrum gives the partial correlation significance
# =========================================================================

out_dir = './results_correlations'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

results = []

for clau_col in claustrum_cols:
    for roi_col in roi_cols:
        # Sanitize column names for formula (hyphens not allowed)
        safe_roi = roi_col.replace('-', '_')
        safe_clau = clau_col.replace('-', '_')
        df[safe_roi] = df[roi_col]
        df[safe_clau] = df[clau_col]

        formula = f"{safe_roi} ~ {safe_clau} + age + Gender_bin + measure_eTIV"

        try:
            model = smf.mixedlm(
                formula,
                data=df,
                groups=df["subj_id"],
            )
            result = model.fit(reml=False)

            coef = result.fe_params[safe_clau]
            se = result.bse_fe[safe_clau]
            z_val = result.tvalues[safe_clau]
            p_val = result.pvalues[safe_clau]

            # Approximate partial r from z and N
            n = len(df)
            partial_r = z_val / np.sqrt(z_val**2 + n)

            results.append({
                'claustrum': clau_col,
                'roi': roi_col,
                'coefficient': coef,
                'std_error': se,
                'z_value': z_val,
                'p_value': p_val,
                'partial_r': partial_r,
            })
            print(f"  {clau_col} ~ {roi_col}: r={partial_r:.3f}, p={p_val:.4f}")

        except Exception as e:
            print(f"  {clau_col} ~ {roi_col}: FAILED ({e})")
            results.append({
                'claustrum': clau_col,
                'roi': roi_col,
                'coefficient': np.nan,
                'std_error': np.nan,
                'z_value': np.nan,
                'p_value': np.nan,
                'partial_r': np.nan,
            })

results_df = pd.DataFrame(results)

# FDR correction
from statsmodels.stats.multitest import multipletests
valid_mask = ~results_df['p_value'].isna()
if valid_mask.sum() > 0:
    _, p_corr, _, _ = multipletests(results_df.loc[valid_mask, 'p_value'], method='fdr_bh')
    results_df.loc[valid_mask, 'p_fdr'] = p_corr

results_df.to_csv(os.path.join(out_dir, 'partial_correlations.csv'), index=False)

print("\n" + "=" * 70)
print("Partial Correlations (FDR-corrected)")
print("=" * 70)
print(results_df[['claustrum', 'roi', 'partial_r', 'p_value', 'p_fdr']].to_string(index=False))

# =========================================================================
# Plot: heatmap of partial correlations
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

for idx, clau_col in enumerate(claustrum_cols):
    ax = axes[idx]
    subset = results_df[results_df['claustrum'] == clau_col].copy()

    # Short names for display
    subset['roi_short'] = subset['roi'].str.replace('subcort_', '').str.replace('cort_', '').str.replace('Left-', 'L ').str.replace('Right-', 'R ').str.replace('_lh_', ' L ').str.replace('_rh_', ' R ')

    colors = ['firebrick' if p < 0.05 else 'steelblue' for p in subset['p_fdr']]
    edge_colors = ['black' if p < 0.05 else 'none' for p in subset['p_fdr']]

    y_pos = np.arange(len(subset))
    bars = ax.barh(y_pos, subset['partial_r'], color=colors, edgecolor=edge_colors, linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subset['roi_short'], fontsize=9)
    ax.set_xlabel('Partial r', fontsize=12)
    ax.set_title(clau_col.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()

    # Add significance markers
    for i, (r, p) in enumerate(zip(subset['partial_r'], subset['p_fdr'])):
        if p < 0.05:
            ax.text(r + 0.005 * np.sign(r), i, '*', fontsize=14, va='center', fontweight='bold')

fig.suptitle('Partial Correlations: Claustrum vs Basal Ganglia / Cingulate\n(controlling for age, sex, eTIV; * = p_FDR < 0.05)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'partial_correlations_heatmap.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'partial_correlations_heatmap.eps'), bbox_inches='tight')
print(f"\nFigures saved to {out_dir}/")
