"""
fit_opt_model.py - Port of fitOptModel.m

Fits mixed-effect models of increasing orders and selects the best model
according to BIC. Tests group and interaction effects with likelihood
ratio tests.
"""

import numpy as np
from .estimate_model import estimate_model, EstimatedModel
from .likelihood_ratio_test import likelihood_ratio_test


def fit_opt_model(input_data, opts):
    """
    Fit optimal mixed-effect models for each data column.

    Parameters
    ----------
    input_data : dict
        Dictionary with keys:
        - 'subj_id': array of subject IDs
        - 'age': array of ages
        - 'grouping': grouping variable (None or array)
        - 'data': matrix (#obs x #models) of response variables
        - 'cov': covariate matrix or None
    opts : dict
        Options with keys:
        - 'orders': list of model orders to test (e.g. [0,1,2,3])
        - 'm_type': 'intercept', 'slope', or 'glm'
        - 'alpha': significance level (default 0.05)
        - 'vert_id': column indices of data to analyze (default: all)
        - 'model_names': list of model names (default: column indices)

    Returns
    -------
    list of EstimatedModel
        Vector of fitted optimal models.
    """
    # Defaults
    alpha = opts.get('alpha', 0.05)
    m_type = opts.get('m_type', 'intercept')
    orders = opts['orders']
    # data = np.asarray(input_data['data'], dtype=float)
    data = input_data['data']
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    vert_id = opts.get('vert_id', list(range(data.shape[1])))
    model_names = opts.get('model_names', [str(v) for v in vert_id])

    # Center covariates
    cov = input_data.get('cov')
    if cov is not None and len(cov) > 0:
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        cov = cov - np.mean(cov, axis=0)
    else:
        cov = None

    subj_id = np.asarray(input_data['subj_id']).flatten()
    age = np.asarray(input_data['age'], dtype=float).flatten()
    grouping = input_data.get('grouping')
    if grouping is not None:
        grouping = np.asarray(grouping, dtype=float)
        if grouping.ndim == 1:
            grouping = grouping.reshape(-1, 1)

    n_models = len(vert_id)
    print(f'n_models: {n_models}')
    out_model_vect = [None] * n_models

    for im in range(n_models):
        iv = vert_id[im]
        print(f"\nModel {iv} : ", end="")
        data_vect = data[:, iv]
        print(f'data_vect: {data_vect.shape}')
        # Handle missing data
        if np.any(np.isnan(data_vect)):
            print("Warning: data vector has NaN values, running without missing data points")
        data_id = ~np.isnan(data_vect)

        # Skip if all zeros
        if not np.any(data_vect[data_id]):
            out_model_vect[im] = EstimatedModel(
                input=None, order=0, design_matrix=None,
                design_vars=[], beta=None, rand_cov=None, stats=None,
                m_name=model_names[im],
            )
            continue

        # Subset data
        sid = subj_id[data_id]
        grp = grouping[data_id] if grouping is not None else None
        ag = age[data_id]
        cv = cov[data_id] if cov is not None else None
        dv = data_vect[data_id]

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
        n_unique_groups = len(np.unique(grouping)) if grouping is not None else 1
        if n_unique_groups > 1:
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
