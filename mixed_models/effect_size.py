"""
effect_size.py - Port of GroupCalculationEffect.m and InterCalculationEffect.m

Extracts and tabulates effect sizes for group and interaction effects.
"""

import pandas as pd


def group_calculation_effect(out_model_vect):
    """
    Compile a table of group effect sizes across models.

    Port of GroupCalculationEffect.m.

    Parameters
    ----------
    out_model_vect : list of EstimatedModel
        Fitted models.

    Returns
    -------
    pd.DataFrame or None
        Table with Chi-square statistics, degrees of freedom, and p-values.
    """
    rows = []
    for im, model in enumerate(out_model_vect):
        if model.group_effect is not None:
            rows.append({
                'Model number': im + 1,
                'Chi square statistic': model.group_effect['Chi2'],
                'Degrees of freedom': model.group_effect['dof_diff'],
                'p-value': model.group_effect['p'],
            })

    if rows:
        return pd.DataFrame(rows)
    return None


def inter_calculation_effect(out_model_vect):
    """
    Compile a table of interaction effect sizes across models.

    Port of InterCalculationEffect.m.

    Parameters
    ----------
    out_model_vect : list of EstimatedModel
        Fitted models.

    Returns
    -------
    pd.DataFrame or None
        Table with Chi-square statistics, degrees of freedom, and p-values.
    """
    rows = []
    for im, model in enumerate(out_model_vect):
        if model.inter_effect is not None:
            rows.append({
                'Model number': im + 1,
                'Chi square statistic': model.inter_effect['Chi2'],
                'Degrees of freedom': model.inter_effect['dof_diff'],
                'p-value': model.inter_effect['p'],
            })

    if rows:
        return pd.DataFrame(rows)
    return None
