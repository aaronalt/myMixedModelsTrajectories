"""
results.py - Port of plotModelsAndSaveResults.m

Saves result tables (as pandas DataFrames) and generates publication-quality figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plotting import plot_model, plot_residuals


def plot_models_and_save_results(out_model_vect, plot_opts, save_results=0, out_dir=None):
    """
    Save models to files and generate figures.

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

    Returns
    -------
    pd.DataFrame or None
        The results table, if save_results > 0.
    """
    if save_results and out_dir is None:
        out_dir = os.getcwd()
        print(f"Warning: no output directory specified, saving to {out_dir}")

    plotting = plot_opts.get('plotting', True)

    # --- Build results table as a DataFrame ---
    rows = []
    for model in out_model_vect:
        if model.stats is None:
            rows.append({'model_name': model.m_name})
            continue

        row = {'model_name': model.m_name, 'model_order': model.order}

        if model.group_effect is not None:
            row['p_val_group'] = model.group_effect['p']

        if model.order > 0 and model.inter_effect is not None:
            row['p_val_interaction'] = model.inter_effect['p']

        # Full model betas — use named index from the Series
        for var_name, val in model.beta.items():
            row[f'full_beta_{var_name}'] = val

        # No-group reduced model betas
        if model.group_effect is not None:
            for var_name, val in model.group_effect['reduced_model'].beta.items():
                row[f'noGroup_beta_{var_name}'] = val

        # No-interaction reduced model betas
        if model.order > 0 and model.inter_effect is not None:
            for var_name, val in model.inter_effect['reduced_model'].beta.items():
                row[f'noInteraction_beta_{var_name}'] = val

        rows.append(row)

    result_table = pd.DataFrame(rows)

    # --- Save ---
    if save_results:
        os.makedirs(out_dir, exist_ok=True)

        if len(out_model_vect) > 1:
            res_file = os.path.join(
                out_dir,
                f"resultTable_{out_model_vect[0].m_name}_to_{out_model_vect[-1].m_name}")
        else:
            res_file = os.path.join(out_dir, f"resultTable_{out_model_vect[0].m_name}")

        result_table.to_csv(f"{res_file}.csv", sep='\t', index=False)
        print(f"Results saved to {res_file}.csv")

    # --- Plot and save figures ---
    if plotting:
        for model in out_model_vect:
            if model.stats is not None:
                fig, plotted_model = plot_model(model, plot_opts)
                fig2, residuals = plot_residuals(model)

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

    return result_table
