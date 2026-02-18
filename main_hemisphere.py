"""
main_hemisphere.py

Fits a mixed-effects model with hemisphere as a within-subject factor.
Data is reshaped to long format (one row per subject-timepoint-hemisphere).
eTIV and sex are regressed out beforehand using compute_residuals.

Model: volume ~ age * grouping * hemisphere
Random effects: random intercept + random slope for age per subject
"""

import numpy as np
import pandas as pd
import matplotlib
import os

from functions.compute_residuals import compute_residuals

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

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

df['clau_lh'] = pd.to_numeric(df['clau_lh'], errors='coerce')
df['clau_rh'] = pd.to_numeric(df['clau_rh'], errors='coerce')

# Drop missing values
df = df.dropna(subset=['grouping', 'Gender_bin', 'measure_eTIV', 'age', 'clau_lh', 'clau_rh']).reset_index(drop=True)

# Regress out eTIV and sex, add back mean to preserve original scale
covariates = [df['Gender_bin'], df['measure_eTIV']]
original_means = df[['clau_lh', 'clau_rh']].mean()
residuals, beta = compute_residuals(df[['clau_lh', 'clau_rh']], covariates)
df[['clau_lh', 'clau_rh']] = residuals + original_means.values

# Remove outlier and restrict age
df = df[~((df['age'] < 25) & (df['clau_lh'] < 500))].reset_index(drop=True)
df = df[df['age'] <= 35].reset_index(drop=True)

# =========================================================================
# Reshape to long format: one row per subject-timepoint-hemisphere
# =========================================================================

df_lh = df[['subj_id', 'age', 'grouping', 'clau_lh']].copy()
df_lh['hemisphere'] = 0
df_lh = df_lh.rename(columns={'clau_lh': 'volume'})

df_rh = df[['subj_id', 'age', 'grouping', 'clau_rh']].copy()
df_rh['hemisphere'] = 1
df_rh = df_rh.rename(columns={'clau_rh': 'volume'})

df_long = pd.concat([df_lh, df_rh], ignore_index=True)
df_long = df_long.sort_values(['subj_id', 'age', 'hemisphere']).reset_index(drop=True)

print(f"Long-format shape: {df_long.shape}")
print(f"Subjects: {df_long['subj_id'].nunique()}")
print(f"Observations per hemisphere: LH={len(df_lh)}, RH={len(df_rh)}")
print(df_long.head(10))

# =========================================================================
# Fit mixed-effects model: volume ~ age * grouping * hemisphere
# =========================================================================

model = smf.mixedlm(
    "volume ~ age * grouping * hemisphere",
    data=df_long,
    groups=df_long["subj_id"],
    re_formula="~age",
)

result = model.fit(reml=False)
print("\n" + "=" * 70)
print("Model: volume ~ age * grouping * hemisphere")
print("Random effects: ~age | subj_id")
print("=" * 70)
print(result.summary())

# =========================================================================
# Save results
# =========================================================================

out_dir = './results_hemisphere'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Save summary to text file
with open(os.path.join(out_dir, 'hemisphere_model_summary.txt'), 'w') as f:
    f.write(str(result.summary()))

# Save fixed effects to CSV
fe = pd.DataFrame({
    'estimate': result.fe_params,
    'std_error': result.bse_fe,
    'z_value': result.tvalues[:len(result.fe_params)],
    'p_value': result.pvalues[:len(result.fe_params)],
})
fe.to_csv(os.path.join(out_dir, 'hemisphere_model_fixed_effects.csv'))

print(f"\nResults saved to {out_dir}/")
print("\nFixed effects:")
print(fe)

# =========================================================================
# Plot: 2x1 subplot (LH and RH), with group trajectories and spaghetti lines
# =========================================================================

fe_params = result.fe_params
age_range = np.linspace(df_long['age'].min(), df_long['age'].max(), 200)

# Build predicted curves for each group x hemisphere combination
def predict_curve(age_vec, grouping_val, hemi_val):
    return (fe_params['Intercept']
            + fe_params['age'] * age_vec
            + fe_params['grouping'] * grouping_val
            + fe_params['age:grouping'] * age_vec * grouping_val
            + fe_params['hemisphere'] * hemi_val
            + fe_params['age:hemisphere'] * age_vec * hemi_val
            + fe_params['grouping:hemisphere'] * grouping_val * hemi_val
            + fe_params['age:grouping:hemisphere'] * age_vec * grouping_val * hemi_val)

colors = {'HC': 'steelblue', '22q': 'firebrick'}
hemi_labels = {0: 'Left Hemisphere', 1: 'Right Hemisphere'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for hemi_idx, ax in enumerate(axes):
    hemi_data = df_long[df_long['hemisphere'] == hemi_idx]

    # Spaghetti lines per subject, colored by group
    for grp_val, grp_label, color in [(0.0, 'HC', colors['HC']), (1.0, '22q', colors['22q'])]:
        grp_data = hemi_data[hemi_data['grouping'] == grp_val]

        # Individual subject lines
        for sid in grp_data['subj_id'].unique():
            subj = grp_data[grp_data['subj_id'] == sid].sort_values('age')
            ax.plot(subj['age'], subj['volume'], '-', color=color, alpha=0.15, linewidth=0.8)

        # Data points
        ax.plot(grp_data['age'], grp_data['volume'], '.', color=color, alpha=0.3, markersize=6)

        # Fitted trajectory
        y_pred = predict_curve(age_range, grp_val, hemi_idx)
        ax.plot(age_range, y_pred, '-', color=color, linewidth=3, label=grp_label)

    ax.set_title(hemi_labels[hemi_idx], fontsize=14, fontweight='bold')
    ax.set_xlabel('Age', fontsize=12)
    ax.legend(fontsize=11)

axes[0].set_ylabel('Claustrum Volume (mm³)', fontsize=12)

fig.suptitle('Claustrum Volume ~ Age × Group × Hemisphere', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'hemisphere_model_plot.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'hemisphere_model_plot.eps'), bbox_inches='tight')
print(f"Figures saved to {out_dir}/hemisphere_model_plot.png")
