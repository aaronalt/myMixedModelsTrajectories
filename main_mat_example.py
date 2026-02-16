"""
main_mat_example.py - Port of main_mat_example.m

Example script for fitting mixed-effect model trajectories
using data from a .mat file.
"""

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; remove this line for interactive use
import matplotlib.pyplot as plt

from mixed_models import (
    fit_opt_model,
    fdr_correct,
    plot_models_and_save_results,
    group_calculation_effect,
    inter_calculation_effect,
)

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
# Set up all necessary options here
# =========================================================================

# --- Load .mat file ---
input_data_file = 'exampleData.mat'
mat = sio.loadmat(input_data_file)

input_data = {
    'subj_id': mat['within_subj'].flatten(),
    'age': mat['age'].flatten().astype(float),
    'grouping': mat['diagnosis'].astype(float),  # 0/1 for 2 groups
    'data': mat['lh_thickness'].astype(float),
    'cov': mat['sex'].astype(float),
}

# --- Model estimation options ---
opts = {
    'orders': [0, 1, 2, 3],
    'm_type': 'slope',
}

# Vertex IDs (columns of lh_thickness to analyze)
vert_id = [2000, 3600, 4000]
opts['vert_id'] = vert_id

# --- Model plotting options ---
out_dir = './results_mat_fdrcorr'
save_results = 2

plot_opts = {
    'leg_txt': ['HC', 'Pat'],
    'x_label': 'age',
    'y_label': 'cortical thickness',
    'plot_ci': True,
    'plot_type': 'redInter',
    'fig_size': (7.3, 4.3),  # approximate MATLAB [440 488 525 310] in inches
}


# =========================================================================
# Execute
# =========================================================================

# --- Run model fitting ---
opts['model_names'] = [str(v) for v in vert_id]
opts['alpha'] = 0.05

out_model_vect = fit_opt_model(input_data, opts)
out_model_vect_corr = fdr_correct(out_model_vect, opts['alpha'])

# --- Plot and save ---
plot_opts['n_cov'] = 1  # sex covariate
plot_models_and_save_results(out_model_vect_corr, plot_opts, save_results, out_dir)

# --- Effect sizes ---
effect_size_group = group_calculation_effect(out_model_vect)
print("\nGroup Effect Sizes:")
print(effect_size_group)

effect_size_inter = inter_calculation_effect(out_model_vect)
print("\nInteraction Effect Sizes:")
print(effect_size_inter)
