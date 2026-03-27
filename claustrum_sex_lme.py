import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("all_fs_volumes.csv")
df = df[df['Diagnosis_bin'] == 0] # Filter for only HC/22q per loop

# TIV-normalize (x1000 for interpretable scale)
df["clau_lh_norm"] = df["clau_lh_Volume_mm3"] / df["measure_eTIV"] * 1000
df["clau_rh_norm"] = df["clau_rh_Volume_mm3"] / df["measure_eTIV"] * 1000
df["clau_avg_norm"] = (df["clau_lh_norm"] + df["clau_rh_norm"]) / 2

# Averaged absolute
df["clau_avg_vol"] = (df["clau_lh_Volume_mm3"] + df["clau_rh_Volume_mm3"]) / 2

# Fix Gender_bin: enforce 0=Female, 1=Male from Gender column
df["Gender_bin"] = df["Gender"].map({"Female": 0, "Male": 1})

# Drop missing
df = df.dropna(subset=["Gender_bin", "Age", "measure_eTIV"])
df["Subject_ID"] = df["Subject_ID"].astype(str)

print(f"N scans: {len(df)}  |  N subjects: {df['Subject_ID'].nunique()}")
print(f"Sex distribution: {df.groupby('Gender')['Subject_ID'].nunique().to_dict()}\n")

# df['Gender_bin'] = df['Gender_bin'].map({0.0: 0.0, 1.1: 1.1}).astype('category')

def run_lme(outcome, label, data):
    m = smf.mixedlm(
        f"{outcome} ~ C(Gender_bin) + Age",
        data=data.dropna(subset=[outcome]),
        groups=data.dropna(subset=[outcome])["Subject_ID"]
    )
    r = m.fit(reml=True)
    coef = r.params.filter(like='Gender_bin').iloc[0]
    se   = r.bse.filter(like='Gender_bin').iloc[0]
    p    = r.pvalues.filter(like='Gender_bin').iloc[0]
    direction = "Male > Female" if coef > 0 else "Female > Male"
    print(f"  {label:<35} coef={coef:+.4f}  SE={se:.4f}  p={p:.4f}  ({direction})")
    return r

# --- Model A: TIV-normalized ---
print("=" * 70)
print("MODEL A: TIV-normalized claustrum ~ sex + age + (1|subject)")
print("=" * 70)
run_lme("clau_avg_norm", "Averaged claustrum", df)
run_lme("clau_rh_norm",  "Right claustrum",    df)
run_lme("clau_lh_norm",  "Left claustrum",     df)

# --- Model B: Absolute volume ---
print()
print("=" * 70)
print("MODEL B: Absolute claustrum ~ sex + age + (1|subject)")
print("=" * 70)
run_lme("clau_avg_vol",         "Averaged claustrum", df)
run_lme("clau_rh_Volume_mm3",   "Right claustrum",    df)
run_lme("clau_lh_Volume_mm3",   "Left claustrum",     df)
