import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
})


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

    # --- Age-binned hemisphere effect ---
    print("\n" + "=" * 60)
    print(f"Age-binned hemisphere effect - {diagnosis_group}")
    print("=" * 60)

    age_bins = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35)]
    bin_results = []

    for lo, hi in age_bins:
        bin_label = f"{lo}-{hi}"
        win = long[(long['Age'] >= lo) & (long['Age'] < hi)]
        n_subj = win['Subject_ID'].nunique()
        n_obs = len(win)

        if n_subj < 10:
            bin_results.append({
                'age_bin': bin_label, 'bin_mid': (lo + hi) / 2,
                'n_subj': n_subj, 'n_obs': n_obs,
                'coef': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan, 'pval': np.nan,
            })
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mw = smf.mixedlm(
                    "clau_norm ~ C(hemisphere) + C(Gender_bin)",
                    data=win, groups=win["Subject_ID"]
                ).fit(reml=True)
            p = 'C(hemisphere)[T.LH]'
            ci = mw.conf_int().loc[p]
            bin_results.append({
                'age_bin': bin_label, 'bin_mid': (lo + hi) / 2,
                'n_subj': n_subj, 'n_obs': n_obs,
                'coef': mw.fe_params[p], 'ci_lo': ci[0], 'ci_hi': ci[1],
                'pval': mw.pvalues[p],
            })
        except Exception:
            bin_results.append({
                'age_bin': bin_label, 'bin_mid': (lo + hi) / 2,
                'n_subj': n_subj, 'n_obs': n_obs,
                'coef': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan, 'pval': np.nan,
            })

    bin_df = pd.DataFrame(bin_results)
    bin_df['group'] = diagnosis_group
    bin_df.to_csv(f'claustrum_hemisphere_agebins_{diagnosis_group.replace(" ", "_").lower()}.csv',
                  index=False)

    valid = bin_df.dropna(subset=['coef'])
    for _, row in valid.iterrows():
        star = '***' if row['pval'] < 0.001 else '**' if row['pval'] < 0.01 else \
               '*' if row['pval'] < 0.05 else ''
        print(f"    {row['age_bin']:>5s} yr (n={row['n_subj']:3.0f}): "
              f"coef={row['coef']:+.4f}, p={row['pval']:.4f} {star}")

    # Plot age-binned results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={'hspace': 0.08})

    for _, row in valid.iterrows():
        color = '#A23B72' if row['pval'] < 0.05 else '#cccccc'
        ax1.errorbar(row['bin_mid'], row['coef'],
                     yerr=[[row['coef'] - row['ci_lo']], [row['ci_hi'] - row['coef']]],
                     fmt='o', color=color, capsize=5, capthick=2,
                     markersize=9, linewidth=2, zorder=5)
        star = '***' if row['pval'] < 0.001 else '**' if row['pval'] < 0.01 else \
               '*' if row['pval'] < 0.05 else ''
        if star:
            ax1.text(row['bin_mid'], row['ci_hi'] + 0.005, star,
                     ha='center', fontsize=11, fontweight='bold', color='#A23B72')

    ax1.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Hemisphere coef\n(LH vs RH, TIV-norm)', fontsize=11)
    ax1.set_title(f'Hemisphere Asymmetry by Age Bin — {diagnosis_group}\n'
                  f'Pink = p<.05 | Grey = n.s.',
                  fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.15)

    ax2.bar(valid['bin_mid'], valid['n_subj'], color='#2E86AB', alpha=0.6, width=4)
    for _, row in valid.iterrows():
        ax2.text(row['bin_mid'], row['n_subj'] + 1, f"{int(row['n_subj'])}",
                 ha='center', va='bottom', fontsize=8, color='#2E86AB', fontweight='bold')
    ax2.set_xlabel('Age (years)', fontsize=11)
    ax2.set_ylabel('N subj', fontsize=11)
    ax2.set_xticks([b['bin_mid'] for b in bin_results])
    ax2.set_xticklabels([b['age_bin'] for b in bin_results])
    ax2.grid(alpha=0.15)

    plt.tight_layout()
    fname = f'claustrum_hemisphere_agebins_{diagnosis_group.replace(" ", "_").lower()}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")
