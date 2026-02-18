"""
prepare_mat_files.py

Cleans the FreeSurfer CSV and exports separate .mat files for MATLAB:
  1. Remove NaNs
  2. Filter age <= 35, remove outlier (age < 25 & clau_lh < 500)
  3. Sort by subject, then by age within each subject
  4. Compute per-subject means and deltas for age and claustrum volumes
  5. Regress out eTIV and sex from brain volumes
  6. Save separate .mat files:
     - demographics.mat
     - claustrum_volumes.mat
     - brain_volumes.mat
"""

import numpy as np
import pandas as pd
from scipy.io import savemat
import os

from functions.compute_residuals import compute_residuals

# =========================================================================
# 1. Load and clean
# =========================================================================

df = pd.read_csv('all_fs_volumes.csv')

df = df.rename(columns={
    'Subject_ID': 'subj_id',
    'Age': 'age',
    'clau_lh_Volume_mm3': 'clau_lh',
    'clau_rh_Volume_mm3': 'clau_rh',
    'Diagnosis_bin': 'grouping',
})

# Coerce claustrum to numeric
df['clau_lh'] = pd.to_numeric(df['clau_lh'], errors='coerce')
df['clau_rh'] = pd.to_numeric(df['clau_rh'], errors='coerce')

# Drop rows missing key columns
df = df.dropna(subset=['grouping', 'Gender_bin', 'measure_eTIV', 'age',
                        'clau_lh', 'clau_rh']).reset_index(drop=True)

# Remove outlier and cap age
df = df[~((df['age'] < 25) & (df['clau_lh'] < 500))].reset_index(drop=True)
df = df[df['age'] <= 35].reset_index(drop=True)

print(f"After cleaning: {len(df)} observations, {df['subj_id'].nunique()} subjects")

# =========================================================================
# 2. Sort by subject, then by age within each subject
# =========================================================================

df = df.sort_values(['subj_id', 'age']).reset_index(drop=True)

# =========================================================================
# 3. Compute per-subject means and deltas
# =========================================================================

subj_means = df.groupby('subj_id').agg(
    mean_age=('age', 'mean'),
    mean_clau_lh=('clau_lh', 'mean'),
    mean_clau_rh=('clau_rh', 'mean'),
).reset_index()

df = df.merge(subj_means, on='subj_id', how='left')

df['delta_age'] = df['age'] - df['mean_age']
df['delta_clau_lh'] = df['clau_lh'] - df['mean_clau_lh']
df['delta_clau_rh'] = df['clau_rh'] - df['mean_clau_rh']

# =========================================================================
# 4. Identify brain volume columns (exclude demographics, eTIV, Mask, Seg)
# =========================================================================

exclude_patterns = ['eTIV', 'MaskVol', 'BrainSeg']
brain_cols = [c for c in df.columns
              if (c.startswith('measure_') or c.startswith('subcort_') or c.startswith('cort_'))
              and not any(pat in c for pat in exclude_patterns)]

# Coerce all brain volume columns to numeric
for col in brain_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where any brain volume is NaN
df = df.dropna(subset=brain_cols).reset_index(drop=True)
print(f"After dropping NaN brain volumes: {len(df)} observations")

# =========================================================================
# 5. Regress out eTIV and sex from brain volumes
# =========================================================================

covariates = [df['Gender_bin'].values, df['measure_eTIV'].values]
brain_data = df[brain_cols].values

original_means = df[brain_cols].mean().values
residuals, _ = compute_residuals(brain_data, covariates)
brain_residualized = residuals + original_means

# Also residualize claustrum
clau_data = df[['clau_lh', 'clau_rh']].values
clau_means = df[['clau_lh', 'clau_rh']].mean().values
clau_resid, _ = compute_residuals(clau_data, covariates)
clau_residualized = clau_resid + clau_means

# Recompute means/deltas on residualized claustrum
df['clau_lh_resid'] = clau_residualized[:, 0]
df['clau_rh_resid'] = clau_residualized[:, 1]

subj_means_resid = df.groupby('subj_id').agg(
    mean_clau_lh_resid=('clau_lh_resid', 'mean'),
    mean_clau_rh_resid=('clau_rh_resid', 'mean'),
).reset_index()
df = df.merge(subj_means_resid, on='subj_id', how='left')

df['delta_clau_lh_resid'] = df['clau_lh_resid'] - df['mean_clau_lh_resid']
df['delta_clau_rh_resid'] = df['clau_rh_resid'] - df['mean_clau_rh_resid']

# =========================================================================
# 6. Save .mat files
# =========================================================================

out_dir = './mat_files'
os.makedirs(out_dir, exist_ok=True)

# Helper: ensure column vector (Nx1) for MATLAB
def col(arr):
    return np.asarray(arr).reshape(-1, 1)

# --- demographics.mat ---
savemat(os.path.join(out_dir, 'demographics.mat'), {
    'subj_id': col(df['subj_id']),
    'age': col(df['age']),
    'mean_age': col(df['mean_age']),
    'delta_age': col(df['delta_age']),
    'sex': col(df['Gender_bin']),
    'group': col(df['grouping']),
    'diagnosis': np.array(df['Diagnosis'].astype(str).tolist(), dtype=object).reshape(-1, 1),
})

# --- claustrum_volumes.mat ---
savemat(os.path.join(out_dir, 'claustrum_volumes.mat'), {
    'clau_lh': col(df['clau_lh']),
    'clau_rh': col(df['clau_rh']),
    'mean_clau_lh': col(df['mean_clau_lh']),
    'mean_clau_rh': col(df['mean_clau_rh']),
    'delta_clau_lh': col(df['delta_clau_lh']),
    'delta_clau_rh': col(df['delta_clau_rh']),
    'clau_lh_resid': col(df['clau_lh_resid']),
    'clau_rh_resid': col(df['clau_rh_resid']),
    'mean_clau_lh_resid': col(df['mean_clau_lh_resid']),
    'mean_clau_rh_resid': col(df['mean_clau_rh_resid']),
    'delta_clau_lh_resid': col(df['delta_clau_lh_resid']),
    'delta_clau_rh_resid': col(df['delta_clau_rh_resid']),
})

# --- brain_volumes.mat (residualized, no eTIV/Mask/Seg columns) ---
# Single NxR matrix + col_names cell array
clean_names = [c.replace('-', '_') for c in brain_cols]
savemat(os.path.join(out_dir, 'brain_volumes.mat'), {
    'volumes': brain_residualized,  # (749 x n_regions)
    'col_names': np.array(clean_names, dtype=object).reshape(1, -1),
})

# --- Summary ---
print(f"\nSaved to {out_dir}/:")
for f in sorted(os.listdir(out_dir)):
    if f.endswith('.mat'):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:30s} ({size_kb:.0f} KB)")
