"""
fdr_correct.py - Port of fdr_correct.m

Applies Benjamini-Hochberg FDR correction to p-values across models.
"""

import numpy as np
from statsmodels.stats.multitest import multipletests


def fdr_correct(out_model_vect, alpha=0.05):
    """
    Apply FDR correction to group and interaction p-values across models.

    Parameters
    ----------
    out_model_vect : list of EstimatedModel
        Vector of fitted models.
    alpha : float
        Significance level.

    Returns
    -------
    list of EstimatedModel
        Models with corrected p-values and significance flags.
    """
    # Collect p-values
    p_gr = []
    p_int = []
    gr_indices = []
    int_indices = []

    for i, model in enumerate(out_model_vect):
        if model.group_effect is not None:
            p_gr.append(model.group_effect['p'])
            gr_indices.append(i)
        if model.inter_effect is not None:
            p_int.append(model.inter_effect['p'])
            int_indices.append(i)

    # Apply FDR correction to group effect p-values
    if p_gr:
        p_gr = np.array(p_gr)
        h_gr_corr, p_gr_corr, _, _ = multipletests(p_gr, alpha=alpha, method='fdr_bh')
        for idx, model_idx in enumerate(gr_indices):
            out_model_vect[model_idx].group_effect['p'] = p_gr_corr[idx]
            out_model_vect[model_idx].group_effect['h'] = h_gr_corr[idx]

    # Apply FDR correction to interaction effect p-values
    if p_int:
        p_int = np.array(p_int)
        h_int_corr, p_int_corr, _, _ = multipletests(p_int, alpha=alpha, method='fdr_bh')
        for idx, model_idx in enumerate(int_indices):
            out_model_vect[model_idx].inter_effect['p'] = p_int_corr[idx]
            out_model_vect[model_idx].inter_effect['h'] = h_int_corr[idx]

    return out_model_vect
