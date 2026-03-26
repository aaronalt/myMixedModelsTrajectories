import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("all_fs_volumes.csv")

# TIV-normalize claustrum volumes (x1000 for interpretable scale)
df["clau_lh_norm"] = df["clau_lh_Volume_mm3"] / df["measure_eTIV"] * 1000
df["clau_rh_norm"] = df["clau_rh_Volume_mm3"] / df["measure_eTIV"] * 1000

df = df[df['Diagnosis_bin'] == 1] # Filter for only HC
# Reshape to long format: one row per hemisphere per scan
lh = df[["Subject_ID", "session", "Age", "Gender_bin", "clau_lh_norm"]].copy()
lh["hemisphere"] = "LH"
lh = lh.rename(columns={"clau_lh_norm": "clau_norm"})

rh = df[["Subject_ID", "session", "Age", "Gender_bin", "clau_rh_norm"]].copy()
rh["hemisphere"] = "RH"
rh = rh.rename(columns={"clau_rh_norm": "clau_norm"})

long = pd.concat([lh, rh], ignore_index=True)
long = long.dropna(subset=["clau_norm", "hemisphere", "Gender_bin", "Age"])

# RH as reference for hemisphere, consistent with the other study
long["hemisphere"] = pd.Categorical(long["hemisphere"], categories=["RH", "LH"])
long["Gender_bin"] = pd.Categorical(long["Gender_bin"])

print(f"N rows (hemisphere x scan): {len(long)}")
print(f"N unique subjects: {long['Subject_ID'].nunique()}\n")

# --- Model 1: TIV-normalized claustrum ~ hemisphere + sex + age + (1 | subject) ---
print("=" * 60)
print("Model 1: TIV-normalized claustrum volume")
print("=" * 60)
m1 = smf.mixedlm(
    "clau_norm ~ C(hemisphere) + C(Gender_bin) + Age",
    data=long,
    groups=long["Subject_ID"]
)
r1 = m1.fit(reml=True)
print(r1.summary())

# --- Model 2: Absolute claustrum ~ hemisphere + sex + age + (1 | subject) ---
lh_abs = df[["Subject_ID", "session", "Age", "Gender_bin", "clau_lh_Volume_mm3"]].copy()
lh_abs["hemisphere"] = "LH"
lh_abs = lh_abs.rename(columns={"clau_lh_Volume_mm3": "clau_vol"})

rh_abs = df[["Subject_ID", "session", "Age", "Gender_bin", "clau_rh_Volume_mm3"]].copy()
rh_abs["hemisphere"] = "RH"
rh_abs = rh_abs.rename(columns={"clau_rh_Volume_mm3": "clau_vol"})

long_abs = pd.concat([lh_abs, rh_abs], ignore_index=True)
long_abs = long_abs.dropna(subset=["clau_vol", "hemisphere", "Gender_bin", "Age"])
long_abs["Subject_ID"] = long_abs["Subject_ID"].astype(str)
long_abs["hemisphere"] = pd.Categorical(long_abs["hemisphere"], categories=["RH", "LH"])
long_abs["Gender_bin"] = pd.Categorical(long_abs["Gender_bin"])

print("\n" + "=" * 60)
print("Model 2: Absolute claustrum volume (no TIV correction)")
print("=" * 60)
m2 = smf.mixedlm(
    "clau_vol ~ C(hemisphere) + C(Gender_bin) + Age",
    data=long_abs,
    groups=long_abs["Subject_ID"]
)
r2 = m2.fit(reml=True)
print(r2.summary())
