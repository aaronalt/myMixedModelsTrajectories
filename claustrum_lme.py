import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
})

all_results = []

for diagnosis in [0, 1]:
    # Load data
    df = pd.read_csv("all_fs_volumes.csv")

    # TIV-normalize claustrum volumes (x1000 for interpretable scale)
    df["clau_lh_norm"] = df["clau_lh_Volume_mm3"] / df["measure_eTIV"] * 1000
    df["clau_rh_norm"] = df["clau_rh_Volume_mm3"] / df["measure_eTIV"] * 1000

    diagnosis_group = 'Healthy Control' if diagnosis == 0 else '22q11ds'

    df = df[df['Diagnosis_bin'] == diagnosis] # Filter for only HC/22q per loop
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
    print(f"Model 1: TIV-normalized claustrum volume - {diagnosis_group}")
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
    print(f"Model 2: Absolute claustrum volume (no TIV correction) - {diagnosis_group}")
    print("=" * 60)
    m2 = smf.mixedlm(
        "clau_vol ~ C(hemisphere) + C(Gender_bin) + Age",
        data=long_abs,
        groups=long_abs["Subject_ID"]
    )
    r2 = m2.fit(reml=True)
    print(r2.summary())

    for name, model in [("TIV-normalized", r1), ("Absolute", r2)]:
        params = ['C(hemisphere)[T.LH]', 'C(Gender_bin)[T.1.0]', 'Age']
        for p in params:
            coef = model.fe_params[p]
            ci = model.conf_int().loc[p]
            pval = model.pvalues[p]
            all_results.append({
                'group': diagnosis_group,
                'model': name,
                'predictor': p,
                'coef': coef,
                'ci_lo': ci[0],
                'ci_hi': ci[1],
                'pval': pval,
            })

results_df = pd.DataFrame(all_results)
results_df.to_csv("claustrum_lme_results.csv", index=False)

# =========================================================================
# Forest plots: HC vs 22q coefficient comparison
# =========================================================================
pred_labels = {
    'C(hemisphere)[T.LH]': 'Hemisphere (LH vs RH)',
    'C(Gender_bin)[T.1.0]': 'Sex (Male vs Female)',
    'Age': 'Age',
}
group_colors = {'Healthy Control': '#2E86AB', '22q11ds': '#A23B72'}
group_offsets = {'Healthy Control': -0.15, '22q11ds': 0.15}

for model_name in ['TIV-normalized', 'Absolute']:
    sub = results_df[results_df['model'] == model_name]
    predictors = list(pred_labels.keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    for grp in ['Healthy Control', '22q11ds']:
        grp_data = sub[sub['group'] == grp]
        for i, pred in enumerate(predictors):
            row = grp_data[grp_data['predictor'] == pred].iloc[0]
            y = i + group_offsets[grp]
            color = group_colors[grp]
            ax.errorbar(row['coef'], y,
                        xerr=[[row['coef'] - row['ci_lo']],
                              [row['ci_hi'] - row['coef']]],
                        fmt='o', color=color, capsize=4, capthick=2,
                        markersize=8, linewidth=2,
                        label=grp if i == 0 else None)
            star = '***' if row['pval'] < 0.001 else '**' if row['pval'] < 0.01 else \
                   '*' if row['pval'] < 0.05 else ''
            if star:
                ax.text(row['ci_hi'] + (row['ci_hi'] - row['ci_lo']) * 0.1, y,
                        star, va='center', fontsize=11, fontweight='bold', color=color)

    ax.axvline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(predictors)))
    ax.set_yticklabels([pred_labels[p] for p in predictors], fontsize=11)
    ax.set_xlabel('Coefficient (95% CI)', fontsize=11)
    unit = '(TIV-normalized × 1000)' if model_name == 'TIV-normalized' else '(mm³)'
    ax.set_title(f'LME Coefficients — {model_name} Claustrum {unit}\n'
                 f'HC vs 22q11DS', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(axis='x', alpha=0.15)
    ax.invert_yaxis()
    plt.tight_layout()
    fname = f'claustrum_lme_forest_{model_name.lower().replace("-", "_")}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")
