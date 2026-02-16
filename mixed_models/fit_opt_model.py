"""
fit_opt_model.py - Port of fitOptModel.m

Fits mixed-effect models of increasing orders and selects the best model
according to BIC. Tests group and interaction effects with likelihood
ratio tests.

Input data is passed as a pandas DataFrame.
"""

import numpy as np
import pandas as pd
from .estimate_model import estimate_model, EstimatedModel
from .likelihood_ratio_test import likelihood_ratio_test


def fit_opt_model(input_df, opts):
    """
    Fit optimal mixed-effect models for each response column.

    Parameters
    ----------
    input_df : pd.DataFrame
        Long-format DataFrame with columns:
        - 'subj_id': subject identifiers
        - 'age': age / time variable
        - one or more response columns (specified by opts['response_cols'])
        - optional grouping column(s) (specified by opts['group_col'])
        - optional covariate column(s) (specified by opts['cov_cols'])
    opts : dict
        Options with keys:
        - 'orders': list of model orders to test (e.g. [0,1,2,3])
        - 'm_type': 'intercept', 'slope', or 'glm'
        - 'alpha': significance level (default 0.05)
        - 'response_cols': list of column names to model as response variables
        - 'group_col': str or list of str, grouping column name(s) (default None)
        - 'cov_cols': list of str, covariate column names (default [])
        - 'model_names': list of display names (default: response_cols)

    Returns
    -------
    list of EstimatedModel
        Vector of fitted optimal models.
    """
    alpha = opts.get('alpha', 0.05)
    m_type = opts.get('m_type', 'intercept')
    orders = opts['orders']

    response_cols = opts['response_cols']
    model_names = opts.get('model_names', response_cols)

    # Grouping columns
    group_col = opts.get('group_col')
    if group_col is None:
        group_col_list = []
    elif isinstance(group_col, str):
        group_col_list = [group_col]
    else:
        group_col_list = list(group_col)

    # Covariate columns
    cov_cols = opts.get('cov_cols', [])

    # Extract arrays from DataFrame
    subj_id = input_df['subj_id'].values
    age = input_df['age'].values.astype(float)

    if group_col_list:
        grouping = input_df[group_col_list].values.astype(float)
    else:
        grouping = None

    if cov_cols:
        cov = input_df[cov_cols].values.astype(float)
        # Center covariates
        cov = cov - cov.mean(axis=0)
    else:
        cov = None

    n_models = len(response_cols)
    out_model_vect = [None] * n_models

    for im in range(n_models):
        col = response_cols[im]
        print(f"\nModel '{col}' : ", end="")
        data_vect = input_df[col].values.astype(float)

        # Handle missing data
        valid_mask = ~np.isnan(data_vect)
        if not valid_mask.all():
            print("Warning: NaN values found, running without missing data points")

        # Skip if all zeros
        if not np.any(data_vect[valid_mask]):
            out_model_vect[im] = EstimatedModel(
                input=None, order=0, design_matrix=pd.DataFrame(),
                beta=pd.Series(dtype=float), rand_cov=None, stats=None,
                m_name=model_names[im],
            )
            continue

        # Subset to valid observations
        sid = subj_id[valid_mask]
        grp = grouping[valid_mask] if grouping is not None else None
        ag = age[valid_mask]
        cv = cov[valid_mask] if cov is not None else None
        dv = data_vect[valid_mask]

        # Fit models of increasing order, select best by BIC
        for io_idx, io in enumerate(orders):
            print(f"{io} ", end="")
            est_opts = {
                'm_order': io,
                'group_effect': True,
                'inter_effect': True,
                'm_type': m_type,
            }

            try:
                tmp_model = estimate_model(sid, grp, ag, cv, dv, est_opts)
            except Exception as e:
                print(f"\n  Warning: order {io} failed ({e}), skipping")
                continue

            if io_idx == 0 or out_model_vect[im] is None:
                out_model_vect[im] = tmp_model
            elif tmp_model.stats.bic < out_model_vect[im].stats.bic - 2:
                out_model_vect[im] = tmp_model
            else:
                break  # BIC not decreasing enough, stop

        out_model_vect[im].m_name = model_names[im]
        print(f"\nFinal model order: {out_model_vect[im].order}")

        # Test group and interaction effects if >1 group
        has_groups = grouping is not None and len(np.unique(grouping[valid_mask])) > 1
        if has_groups:
            # Likelihood ratio test for significant group effect
            est_opts_no_group = {
                'm_order': out_model_vect[im].order,
                'group_effect': False,
                'inter_effect': False,
                'm_type': m_type,
            }

            try:
                reduced_model = estimate_model(sid, grp, ag, cv, dv, est_opts_no_group)
                dof = len(out_model_vect[im].beta) - len(reduced_model.beta)
                h, p, dof_diff, chi2_stat = likelihood_ratio_test(
                    out_model_vect[im].stats.logl, reduced_model.stats.logl,
                    dof, alpha,
                )
                out_model_vect[im].group_effect = {
                    'h': h, 'p': p, 'dof_diff': dof_diff, 'Chi2': chi2_stat,
                    'reduced_model': reduced_model,
                }
            except Exception as e:
                print(f"  Warning: group effect test failed ({e})")

            # Likelihood ratio test for significant age*group interaction
            if out_model_vect[im].order > 0:
                est_opts_no_inter = {
                    'm_order': out_model_vect[im].order,
                    'group_effect': True,
                    'inter_effect': False,
                    'm_type': m_type,
                }

                try:
                    reduced_model = estimate_model(sid, grp, ag, cv, dv, est_opts_no_inter)
                    dof = len(out_model_vect[im].beta) - len(reduced_model.beta)
                    h, p, dof_diff, chi2_stat = likelihood_ratio_test(
                        out_model_vect[im].stats.logl, reduced_model.stats.logl,
                        dof, alpha,
                    )
                    out_model_vect[im].inter_effect = {
                        'h': h, 'p': p, 'dof_diff': dof_diff, 'Chi2': chi2_stat,
                        'reduced_model': reduced_model,
                    }
                except Exception as e:
                    print(f"  Warning: interaction effect test failed ({e})")

    return out_model_vect
