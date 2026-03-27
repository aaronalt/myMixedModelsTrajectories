import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("all_fs_volumes.csv")

# TIV-normalize claustrum volumes (x1000 for interpretable scale)
df["clau_lh_norm"] = df["clau_lh_Volume_mm3"] / df["measure_eTIV"] * 1000
df["clau_rh_norm"] = df["clau_rh_Volume_mm3"] / df["measure_eTIV"] * 1000

df = df[df['Diagnosis_bin'] == 0] # Filter for only HC/22q per loop

# Reshape to long format: one row per hemisphere per scan
lh = df[["Subject_ID", "session", "Age", "Gender_bin", "clau_lh_norm"]].copy()
lh["hemisphere"] = "LH"
lh = lh.rename(columns={"clau_lh_norm": "clau_norm"})

rh = df[["Subject_ID", "session", "Age", "Gender_bin", "clau_rh_norm"]].copy()
rh["hemisphere"] = "RH"
rh = rh.rename(columns={"clau_rh_norm": "clau_norm"})

long = pd.concat([lh, rh], ignore_index=True)

# Drop rows with missing values in model variables
long = long.dropna(subset=["clau_norm", "hemisphere", "Gender_bin", "Age"])

# hemisphere and sex as categorical
long["hemisphere"] = pd.Categorical(long["hemisphere"], categories=["RH", "LH"])
long["Gender_bin"] = pd.Categorical(long["Gender_bin"])

print(f"N rows (hemisphere x scan): {len(long)}")
print(f"N unique subjects: {long['Subject_ID'].nunique()}\n")

# OLS: TIV-normalized claustrum ~ hemisphere + sex + age
model = smf.ols("clau_norm ~ C(hemisphere) + C(Gender_bin) + Age", data=long)
result = model.fit()

print(result.summary())
