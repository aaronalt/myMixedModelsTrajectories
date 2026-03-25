"""
run_pls_HC_brain.py

Behavioral PLS: claustrum volume developmental slopes vs whole-brain
parcellation slopes in the HC group.

X (seed):   claustrum slopes (LH, RH) — 2 variables
Y (target): brain parcellation slopes (subcort_*, cort_*) — 113 variables

All volumes residualized for eTIV, sex, euler_z before computing slopes.
Within-subject slopes over age for subjects with >= 2 timepoints.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import savemat
from scipy.linalg import svd
import os

from functions.compute_residuals import compute_residuals

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
})

# =========================================================================
# 1. Load and prepare
# =========================================================================

df = pd.read_csv('all_fs_volumes.csv')

df = df.rename(columns={
    'Subject_ID': 'subj_id',
    'Age': 'age',
    'clau_lh_Volume_mm3': 'clau_lh',
    'clau_rh_Volume_mm3': 'clau_rh',
    'Diagnosis_bin': 'grouping',
})

for col in ['clau_lh', 'clau_rh', 'measure_eTIV', 'grouping', 'age',
            'Gender_bin', 'euler_z']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\n--- QC tracking (HC) ---")
print(f"[0] Raw data loaded: {len(df)} observations, {df['subj_id'].nunique()} subjects")

df = df.dropna(subset=['grouping', 'Gender_bin', 'measure_eTIV', 'age',
                        'euler_z', 'clau_lh', 'clau_rh']).reset_index(drop=True)
# df = df[df['age'] <= 35].reset_index(drop=True)
# print(f"[1] After NaN drop + age<=35 filter: {len(df)} observations, {df['subj_id'].nunique()} subjects")

# Remove claustrum outliers (>3 SD)
for col in ['clau_lh', 'clau_rh']:
    mu, sigma = df[col].mean(), df[col].std()
    df = df[(df[col] >= mu - 3 * sigma) & (df[col] <= mu + 3 * sigma)]
df = df.reset_index(drop=True)
print(f"[2] After claustrum outlier removal (>3 SD): {len(df)} observations, {df['subj_id'].nunique()} subjects")

# Identify brain parcellation columns
brain_cols = [c for c in df.columns if c.startswith('subcort_') or c.startswith('cort_')]
drop_prefixes = ['subcort_Left-Lateral-Ventricle', 'subcort_Right-Lateral-Ventricle',
                 'subcort_Left-Inf-Lat-Vent', 'subcort_Right-Inf-Lat-Vent',
                 'subcort_3rd-Ventricle', 'subcort_4th-Ventricle', 'subcort_5th-Ventricle',
                 'subcort_CSF', 'subcort_Left-choroid-plexus', 'subcort_Right-choroid-plexus',
                 'subcort_CC_Anterior', 'subcort_CC_Central', 'subcort_CC_Mid_Anterior',
                 'subcort_CC_Mid_Posterior', 'subcort_CC_Posterior',
                 'subcort_Brain-Stem', 'subcort_WM-hypointensities',
                 'subcort_Left-WM-hypointensities', 'subcort_Right-WM-hypointensities',
                 'subcort_non-WM-hypointensities', 'subcort_Left-non-WM-hypointensities',
                 'subcort_Right-non-WM-hypointensities',
                 'subcort_Optic-Chiasm', 'subcort_Left-vessel', 'subcort_Right-vessel']

brain_cols = [c for c in brain_cols if c not in drop_prefixes]
for col in brain_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN brain volumes
df = df.dropna(subset=brain_cols).reset_index(drop=True)
print(f"[3] After brain NaN drop: {len(df)} observations, {df['subj_id'].nunique()} subjects, {len(brain_cols)} regions")

# Filter to HC
df_HC = df[df['grouping'] == 0].copy().reset_index(drop=True)
print(f"[4] After HC filter: {len(df_HC)} observations, {df_HC['subj_id'].nunique()} subjects")
print(f"Brain parcellations: {len(brain_cols)}")

# =========================================================================
# 2. Residualize all volumes for eTIV, sex, euler_z
# =========================================================================

covariates = [df_HC['Gender_bin'].values, df_HC['measure_eTIV'].values,
              df_HC['euler_z'].values]

# Claustrum
for hemi in ['clau_lh', 'clau_rh']:
    raw = df_HC[hemi].values.copy()
    resid, _ = compute_residuals(raw.reshape(-1, 1), covariates)
    df_HC[f'{hemi}_resid'] = resid.ravel() + raw.mean()

# Brain parcellations
brain_data = df_HC[brain_cols].values
brain_means = brain_data.mean(axis=0)
brain_resid, _ = compute_residuals(brain_data, covariates)
brain_resid += brain_means

resid_brain_cols = [f'{c}_resid' for c in brain_cols]
for i, col in enumerate(resid_brain_cols):
    df_HC[col] = brain_resid[:, i]

print("Residualization complete")

# =========================================================================
# 3. Compute within-subject slopes
# =========================================================================

subj_counts = df_HC['subj_id'].value_counts()
multi_subjs = sorted(subj_counts[subj_counts >= 2].index.tolist())
print(f"[5] Subjects with >= 2 timepoints: {len(multi_subjs)}")


def compute_slope(ages, values):
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan
    if len(np.unique(ages[mask])) < 2:
        return np.nan
    slope, _, _, _, _ = stats.linregress(ages[mask], values[mask])
    return slope


clau_slope_cols = ['clau_lh_resid_slope', 'clau_rh_resid_slope']
brain_slope_cols = [f'{c}_slope' for c in brain_cols]

slopes_data = {col: [] for col in ['subj_id', 'n_tp', 'age_range']
               + clau_slope_cols + brain_slope_cols}

for subj in multi_subjs:
    sub = df_HC[df_HC['subj_id'] == subj]
    ages = sub['age'].values

    slopes_data['subj_id'].append(subj)
    slopes_data['n_tp'].append(len(sub))
    slopes_data['age_range'].append(ages.max() - ages.min())

    # Claustrum slopes
    for hemi, scol in zip(['clau_lh_resid', 'clau_rh_resid'], clau_slope_cols):
        slopes_data[scol].append(compute_slope(ages, sub[hemi].values))

    # Brain region slopes
    for bcol, scol in zip(resid_brain_cols, brain_slope_cols):
        slopes_data[scol].append(compute_slope(ages, sub[bcol].values))

slopes_df = pd.DataFrame(slopes_data)

# Require both claustrum slopes present
n_before = len(slopes_df)
slopes_df = slopes_df.dropna(subset=clau_slope_cols).reset_index(drop=True)
print(f"[6] After dropping missing claustrum slopes: {len(slopes_df)} subjects (removed {n_before - len(slopes_df)})")

# Drop brain regions with >20% missing slopes
n_regions_before = len(brain_slope_cols)
good_brain = []
for col in brain_slope_cols:
    pct_missing = slopes_df[col].isna().mean()
    if pct_missing <= 0.20:
        good_brain.append(col)
    else:
        print(f"  Dropping {col} ({pct_missing*100:.0f}% missing)")
print(f"[7] After dropping regions >20% missing: {len(good_brain)} regions retained (removed {n_regions_before - len(good_brain)})")

# For remaining, require subjects to have >= 80% of brain slopes
n_before = len(slopes_df)
brain_missing = slopes_df[good_brain].isna().sum(axis=1)
slopes_df = slopes_df[brain_missing <= len(good_brain) * 0.2].reset_index(drop=True)
print(f"[8] After dropping subjects >20% missing brain slopes: {len(slopes_df)} subjects (removed {n_before - len(slopes_df)})")

# Impute remaining NaN with column median
total_imputed = slopes_df[good_brain].isna().sum().sum()
for col in good_brain:
    slopes_df[col] = slopes_df[col].fillna(slopes_df[col].median())
print(f"[9] After median imputation: {len(slopes_df)} subjects, {len(good_brain)} regions ({total_imputed} values imputed)")

print(f"\nSubjects for PLS: {len(slopes_df)}")
print(f"  Brain regions: {len(good_brain)}")
print(f"  Mean age range: {slopes_df['age_range'].mean():.1f} yrs")
print(f"  Mean timepoints: {slopes_df['n_tp'].mean():.1f}")

# Z-score all slopes
for col in clau_slope_cols + good_brain:
    mu, sigma = slopes_df[col].mean(), slopes_df[col].std()
    if sigma > 0:
        slopes_df[col] = (slopes_df[col] - mu) / sigma

# Remove slope outliers (>3 SD on z-scored claustrum slopes)
n_before = len(slopes_df)
for col in clau_slope_cols:
    slopes_df = slopes_df[slopes_df[col].abs() <= 3]
slopes_df = slopes_df.reset_index(drop=True)
print(f"[10] After slope outlier removal (>3 SD): {len(slopes_df)} subjects (removed {n_before - len(slopes_df)})")

X = slopes_df[clau_slope_cols].values   # N x 2 (claustrum)
Y = slopes_df[good_brain].values        # N x p (brain regions)

# Clean region labels
region_labels = [c.replace('_resid_slope', '').replace('subcort_', '').replace('cort_', '')
                 for c in good_brain]

print(f"\nX (claustrum): {X.shape}, Y (brain): {Y.shape}")

# =========================================================================
# 4. Behavioral PLS
# =========================================================================

N_PERM = 5000
N_BOOT = 5000


def run_svd(X, Y):
    R = Y.T @ X
    U, S, Vt = svd(R, full_matrices=False)
    V = Vt.T
    return U, S, V, R


def pls_permutation_test(X, Y, S_observed, n_perm):
    n = X.shape[0]
    n_lvs = len(S_observed)
    exceed_count = np.zeros(n_lvs)
    for _ in range(n_perm):
        perm_idx = np.random.permutation(n)
        X_perm = X[perm_idx]
        R_perm = Y.T @ X_perm
        _, S_perm, _ = svd(R_perm, full_matrices=False)
        for lv in range(n_lvs):
            if S_perm[lv] >= S_observed[lv]:
                exceed_count[lv] += 1
    return (exceed_count + 1) / (n_perm + 1)


def pls_bootstrap(X, Y, U_obs, V_obs, n_boot):
    n = X.shape[0]
    n_brain = X.shape[1]
    n_regions = Y.shape[1]
    n_lvs = U_obs.shape[1]

    U_boot = np.zeros((n_boot, n_regions, n_lvs))
    V_boot = np.zeros((n_boot, n_brain, n_lvs))

    for b in range(n_boot):
        boot_idx = np.random.choice(n, size=n, replace=True)
        X_b, Y_b = X[boot_idx], Y[boot_idx]
        R_b = Y_b.T @ X_b
        U_b, _, Vt_b = svd(R_b, full_matrices=False)
        V_b = Vt_b.T

        for lv in range(n_lvs):
            if np.dot(U_b[:, lv], U_obs[:, lv]) < 0:
                U_b[:, lv] *= -1
                V_b[:, lv] *= -1

        U_boot[b] = U_b[:, :n_lvs]
        V_boot[b] = V_b[:, :n_lvs]

    U_se = U_boot.std(axis=0)
    V_se = V_boot.std(axis=0)
    U_bsr = U_obs / np.where(U_se > 0, U_se, 1)
    V_bsr = V_obs / np.where(V_se > 0, V_se, 1)

    return U_bsr, V_bsr, U_boot, V_boot


print("\n" + "=" * 70)
print("BEHAVIORAL PLS: Claustrum slopes vs Brain Region slopes (HC)")
print("=" * 70)

U, S, V, R = run_svd(X, Y)
n_lvs = len(S)

total_var = np.sum(S ** 2)
var_explained = (S ** 2) / total_var * 100

print(f"\nSingular values: {S}")
print(f"Variance explained: {var_explained}")

print(f"\nRunning {N_PERM} permutations...")
p_perm = pls_permutation_test(X, Y, S, N_PERM)

print(f"Running {N_BOOT} bootstraps...")
U_bsr, V_bsr, U_boot, V_boot = pls_bootstrap(X, Y, U, V, N_BOOT)

brain_scores = X @ V
region_scores = Y @ U

# =========================================================================
# 5. Print results
# =========================================================================

print("\n" + "=" * 70)
clau_names = ['LH Claustrum', 'RH Claustrum']

for lv in range(n_lvs):
    print(f"\nLV{lv+1}: singular value = {S[lv]:.3f}, "
          f"p = {p_perm[lv]:.4f}, "
          f"variance = {var_explained[lv]:.1f}%")

    if p_perm[lv] < 0.05:
        print("  *** SIGNIFICANT ***")

    # Claustrum saliences
    print(f"\n  Claustrum saliences (V) and bootstrap ratios:")
    for v in range(2):
        bsr = V_bsr[v, lv]
        sig = '***' if abs(bsr) > 3 else '**' if abs(bsr) > 2 else ''
        print(f"    {clau_names[v]:20s}: salience={V[v,lv]:.3f}, BSR={bsr:.2f} {sig}")

    # Top brain region saliences (|BSR| > 2 or top 15)
    bsr_vals = U_bsr[:, lv]
    bsr_order = np.argsort(-np.abs(bsr_vals))

    reliable = np.sum(np.abs(bsr_vals) >= 2)
    n_show = max(reliable, 15)

    print(f"\n  Top brain region saliences (showing {n_show}, {reliable} with |BSR|>=2):")
    for rank, idx in enumerate(bsr_order[:n_show]):
        bsr = bsr_vals[idx]
        sig = '***' if abs(bsr) > 3 else '**' if abs(bsr) > 2 else ''
        print(f"    {rank+1:2d}. {region_labels[idx]:40s}: sal={U[idx,lv]:.3f}, BSR={bsr:.2f} {sig}")

    r_scores, p_scores = stats.pearsonr(brain_scores[:, lv], region_scores[:, lv])
    print(f"\n  Claustrum-Brain score correlation: r={r_scores:.3f}, p={p_scores:.4f}")

print("\n" + "=" * 70)

# =========================================================================
# 6. Save results
# =========================================================================

out_dir = './results_pls_HC_brain'
os.makedirs(out_dir, exist_ok=True)

# Full CSV
rows = []
for lv in range(n_lvs):
    for idx in range(len(region_labels)):
        rows.append({
            'LV': lv + 1, 'LV_p': p_perm[lv], 'LV_var_pct': var_explained[lv],
            'Variable': region_labels[idx], 'Type': 'brain_region',
            'Salience': U[idx, lv], 'BSR': U_bsr[idx, lv],
        })
    for v, name in enumerate(clau_names):
        rows.append({
            'LV': lv + 1, 'LV_p': p_perm[lv], 'LV_var_pct': var_explained[lv],
            'Variable': name, 'Type': 'claustrum',
            'Salience': V[v, lv], 'BSR': V_bsr[v, lv],
        })

results_csv = pd.DataFrame(rows)
results_csv.to_csv(os.path.join(out_dir, 'pls_brain_HC.csv'), index=False)

savemat(os.path.join(out_dir, 'pls_brain_HC.mat'), {
    'U': U, 'S': S, 'V': V,
    'U_bsr': U_bsr, 'V_bsr': V_bsr,
    'p_perm': p_perm, 'var_explained': var_explained,
    'brain_scores': brain_scores, 'region_scores': region_scores,
    'clau_names': np.array(clau_names, dtype=object).reshape(1, -1),
    'region_names': np.array(region_labels, dtype=object).reshape(1, -1),
    'X': X, 'Y': Y,
})

slopes_df.to_csv(os.path.join(out_dir, 'pls_brain_slopes_HC.csv'), index=False)

# =========================================================================
# 7. Plots
# =========================================================================

for lv in range(n_lvs):
    if p_perm[lv] >= 0.10:
        continue

    bsr_vals = U_bsr[:, lv]
    bsr_order = np.argsort(bsr_vals)

    # --- Bar plot of all region BSRs ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 18),
                              gridspec_kw={'width_ratios': [5, 1]})

    ax = axes[0]
    sorted_bsr = bsr_vals[bsr_order]
    sorted_labels = [region_labels[i] for i in bsr_order]
    colors = ['#A23B72' if b > 2 else '#2E86AB' if b < -2 else '#cccccc'
              for b in sorted_bsr]

    ax.barh(range(len(sorted_bsr)), sorted_bsr, color=colors, edgecolor='none', height=0.8)
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=6)
    ax.axvline(-2, color='red', linestyle='--', alpha=0.5)
    ax.axvline(2, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Bootstrap Ratio', fontsize=11)
    ax.set_title('Brain Region Saliences', fontsize=12, fontweight='bold')

    # Claustrum saliences
    ax2 = axes[1]
    clau_bsr = V_bsr[:, lv]
    colors_c = ['#2E86AB' if abs(b) > 2 else '#cccccc' for b in clau_bsr]
    ax2.bar(range(2), clau_bsr, color=colors_c, edgecolor='white')
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(['LH', 'RH'], fontsize=10)
    ax2.axhline(-2, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(2, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Bootstrap Ratio', fontsize=11)
    ax2.set_title('Claustrum', fontsize=12, fontweight='bold')

    fig.suptitle(f'PLS LV{lv+1}: Claustrum vs Brain Regions (p={p_perm[lv]:.4f}, '
                 f'{var_explained[lv]:.1f}% var) — HC',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'pls_brain_lv{lv+1}_saliences.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # --- Score scatter ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(brain_scores[:, lv], region_scores[:, lv],
               c='#A23B72', alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    z = np.polyfit(brain_scores[:, lv], region_scores[:, lv], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(brain_scores[:, lv].min(), brain_scores[:, lv].max(), 100)
    ax.plot(x_line, p_line(x_line), 'k--', linewidth=2, alpha=0.6)
    r_val, p_val = stats.pearsonr(brain_scores[:, lv], region_scores[:, lv])
    ax.set_xlabel(f'Claustrum Score (LV{lv+1})', fontsize=11)
    ax.set_ylabel(f'Brain Region Score (LV{lv+1})', fontsize=11)
    ax.set_title(f'LV{lv+1} Score Correlation (r={r_val:.3f}, p={p_val:.4f})',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'pls_brain_lv{lv+1}_scores.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # --- Top reliable regions only ---
    reliable_mask = np.abs(bsr_vals) >= 2
    if reliable_mask.sum() > 0:
        reliable_idx = np.where(reliable_mask)[0]
        reliable_idx = reliable_idx[np.argsort(bsr_vals[reliable_idx])]

        fig, ax = plt.subplots(figsize=(10, max(4, len(reliable_idx) * 0.35)))
        r_bsr = bsr_vals[reliable_idx]
        r_labels = [region_labels[i] for i in reliable_idx]
        colors = ['#A23B72' if b > 0 else '#2E86AB' for b in r_bsr]
        ax.barh(range(len(r_bsr)), r_bsr, color=colors, edgecolor='white')
        ax.set_yticks(range(len(r_labels)))
        ax.set_yticklabels(r_labels, fontsize=9)
        ax.axvline(-2, color='red', linestyle='--', alpha=0.4)
        ax.axvline(2, color='red', linestyle='--', alpha=0.4)
        ax.set_xlabel('Bootstrap Ratio', fontsize=11)
        ax.set_title(f'LV{lv+1} Reliable Regions (|BSR| >= 2)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'pls_brain_lv{lv+1}_reliable.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

print(f"\nAll results saved to {out_dir}/")
