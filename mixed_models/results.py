"""
results.py - Port of plotModelsAndSaveResults.m

Saves result tables and generates publication-quality figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plotting import plot_model, plot_residuals


def plot_models_and_save_results(out_model_vect, plot_opts, save_results=0, out_dir=None):
    """
    Save models to files and generate figures.

    Port of plotModelsAndSaveResults.m.

    Parameters
    ----------
    out_model_vect : list of EstimatedModel
        Vector of fitted models.
    plot_opts : dict
        Plotting options (see plot_model).
    save_results : int
        0: no saving, 1: save table only, 2: save table and figures.
    out_dir : str or None
        Output directory.
    """
    if save_results and out_dir is None:
        out_dir = os.getcwd()
        print(f"Warning: no output directory specified, saving to {out_dir}")

    n_cov = plot_opts.get('n_cov', 0)
    plot_type = plot_opts.get('plot_type', 'redGrp')
    plotting = plot_opts.get('plotting', True)
    fig_size = plot_opts.get('fig_size', (8, 5))

    n_mod = len(out_model_vect)

    # --- Save results table ---
    if save_results:
        os.makedirs(out_dir, exist_ok=True)

        rows = []
        for im, model in enumerate(out_model_vect):
            row = {'model_name': model.m_name}

            if model.stats is not None:
                row['model_order'] = model.order

                if model.group_effect is not None:
                    row['p_val_group'] = model.group_effect['p']

                if model.order > 0 and model.inter_effect is not None:
                    row['p_val_interaction'] = model.inter_effect['p']

                # Full model betas
                for ib, var_name in enumerate(model.design_vars):
                    row[f'full_beta_{var_name}'] = model.beta[ib]

                # No-group reduced model betas
                if model.group_effect is not None:
                    red = model.group_effect['reduced_model']
                    for ib, var_name in enumerate(red.design_vars):
                        row[f'noGroup_beta_{var_name}'] = red.beta[ib]

                # No-interaction reduced model betas
                if model.order > 0 and model.inter_effect is not None:
                    red = model.inter_effect['reduced_model']
                    for ib, var_name in enumerate(red.design_vars):
                        row[f'noInteraction_beta_{var_name}'] = red.beta[ib]

            rows.append(row)

        tab = pd.DataFrame(rows)

        # Output file name
        if len(out_model_vect) > 1:
            res_file = os.path.join(out_dir,
                                    f"resultTable_{out_model_vect[0].m_name}_to_{out_model_vect[-1].m_name}")
        else:
            res_file = os.path.join(out_dir, f"resultTable_{out_model_vect[0].m_name}")

        tab.to_csv(f"{res_file}.csv", sep='\t', index=False)
        print(f"Results saved to {res_file}.csv")

    # --- Plot and save figures ---
    if plotting:
        for im, model in enumerate(out_model_vect):
            if model.stats is not None:
                # Main model plot
                fig, plotted_model = plot_model(model, plot_opts)

                # Residual normality plot
                fig2, residuals = plot_residuals(model)

                # Save figures
                if save_results == 2:
                    fig.savefig(os.path.join(out_dir, f"{model.m_name}_{plotted_model}.png"),
                                dpi=300, bbox_inches='tight')
                    fig.savefig(os.path.join(out_dir, f"{model.m_name}_{plotted_model}.eps"),
                                format='eps', bbox_inches='tight')
                    fig2.savefig(os.path.join(out_dir,
                                              f"{model.m_name}_{plotted_model}_ResNormplot.png"),
                                 dpi=300, bbox_inches='tight')
                    print(f"Figures saved for model: {model.m_name}")

                plt.close(fig)
                plt.close(fig2)
