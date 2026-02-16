"""
estimate_model.py - Port of estimateModel.m

Fits a single mixed-effect model of specified order using statsmodels.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


@dataclass
class ModelStats:
    """Statistics from a fitted model (mirrors MATLAB stats struct)."""
    bic: float
    logl: float
    sse: float
    covb: np.ndarray  # covariance of fixed effects
    sebeta: np.ndarray  # standard errors of fixed effects
    iwres: np.ndarray  # individual weighted residuals


@dataclass
class ModelInput:
    """Input data stored with the model."""
    subj_id: np.ndarray
    age: np.ndarray
    grouping: np.ndarray
    data: np.ndarray
    cov: Optional[np.ndarray]


@dataclass
class EstimatedModel:
    """Output of estimate_model (mirrors MATLAB outModel struct)."""
    input: ModelInput
    order: int
    design_matrix: np.ndarray
    design_vars: list
    beta: np.ndarray
    rand_cov: Optional[np.ndarray]
    stats: ModelStats
    m_name: str = ""
    group_effect: Optional[dict] = None
    inter_effect: Optional[dict] = None


def estimate_model(subj_id, grouping, age, cov, data, opts):
    """
    Fit a mixed-effect model of given order.

    Parameters
    ----------
    subj_id : array-like
        Subject IDs (#obs x 1).
    grouping : array-like
        Grouping variable. Empty/None for 1 group, 1 column for 2 groups,
        2 columns for 3 groups. Values should be 0/1.
    age : array-like
        Age vector (#obs x 1).
    cov : array-like or None
        Covariates matrix (#obs x #cov), or None if no covariates.
    data : array-like
        Response variable (#obs x 1).
    opts : dict
        Options with keys:
        - 'm_order' (int): polynomial order
        - 'group_effect' (bool): include group in model (default True)
        - 'inter_effect' (bool): include age*group interaction (default True)
        - 'm_type' (str): 'intercept', 'slope', or 'glm' (default 'intercept')

    Returns
    -------
    EstimatedModel
        Fitted model with parameters, statistics, and design matrix.
    """
    # Convert to numpy arrays
    subj_id = np.asarray(subj_id).flatten()
    age = np.asarray(age, dtype=float).flatten()
    data = np.asarray(data, dtype=float).flatten()

    if grouping is None or (hasattr(grouping, '__len__') and len(grouping) == 0):
        grouping = np.empty((len(subj_id), 0))
    else:
        grouping = np.asarray(grouping, dtype=float)
        if grouping.ndim == 1:
            grouping = grouping.reshape(-1, 1)

    if cov is not None and len(cov) > 0:
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
    else:
        cov = None

    # Defaults
    group_effect = opts.get('group_effect', True)
    inter_effect = opts.get('inter_effect', True)
    m_type = opts.get('m_type', 'intercept')
    m_order = opts['m_order']

    if not group_effect:
        inter_effect = False

    # Determine number of groups
    n_obs = len(subj_id)
    if grouping.shape[1] == 0:
        groups = 1
    else:
        groups = grouping.shape[1] + 1

    if groups == 2:
        unique_vals = np.unique(grouping[:, 0])
        if not np.array_equal(np.sort(unique_vals), np.array([0., 1.])):
            raise ValueError("Error in group specification: expected 0 and 1 values")

    if groups == 3:
        for col in range(2):
            unique_vals = np.unique(grouping[:, col])
            if not np.array_equal(np.sort(unique_vals), np.array([0., 1.])):
                raise ValueError("Error in group specification: expected 0 and 1 values")

    if groups > 3:
        raise ValueError("More than 3 groups found")

    # Build design matrix
    design_matrix = np.ones((n_obs, 1))
    design_vars = ['1']

    if group_effect:
        if groups > 1:
            for i in range(groups - 1):
                design_matrix = np.column_stack([design_matrix, grouping[:, i]])
                design_vars.append(f'grouping_{i + 1}')
        else:
            print("Warning: only one group, no group effect will be taken into account")
            group_effect = False
            inter_effect = False

    if m_order > 0:
        for io in range(1, m_order + 1):
            design_matrix = np.column_stack([design_matrix, age ** io])
            if io == 1:
                design_vars.append('age')
            else:
                design_vars.append(f'age_{io}')

            if inter_effect:
                # age^io * each grouping column
                for i in range(groups - 1):
                    design_matrix = np.column_stack([design_matrix, (age ** io) * grouping[:, i]])
                    if io == 1:
                        design_vars.append(f'age_by_grouping_{i + 1}')
                    else:
                        design_vars.append(f'age{io}_by_grouping_{i + 1}')

    # Add covariates
    if cov is not None:
        design_matrix = np.column_stack([design_matrix, cov])
        for ic in range(cov.shape[1]):
            design_vars.append(f'Covariate {ic + 1}')

    # Store input
    model_input = ModelInput(
        subj_id=subj_id,
        age=age,
        grouping=grouping,
        data=data,
        cov=cov,
    )

    # Fit model
    if m_type == 'glm':
        # OLS (no random effects)
        ols_model = sm.OLS(data, design_matrix)
        result = ols_model.fit()

        beta = result.params
        rand_cov = None
        stats = ModelStats(
            bic=result.bic,
            logl=result.llf,
            sse=result.ssr,
            covb=result.cov_params(),
            sebeta=result.bse,
            iwres=result.resid,
        )
    else:
        # Mixed model
        # Build random effects design matrix
        if m_type == 'slope' and m_order > 0:
            # Random intercept + slope
            if group_effect:
                # Columns: intercept and age (which is column index `groups` in design matrix)
                exog_re = np.column_stack([np.ones(n_obs), age])
            else:
                exog_re = np.column_stack([np.ones(n_obs), age])
        else:
            # Random intercept only
            exog_re = np.ones((n_obs, 1))

        # Use integer subject IDs as groups for MixedLM
        mixed_model = MixedLM(
            endog=data,
            exog=design_matrix,
            groups=subj_id,
            exog_re=exog_re,
        )

        try:
            result = mixed_model.fit(reml=False, maxiter=500)
        except Exception:
            # Fall back to simpler model if convergence fails
            exog_re_simple = np.ones((n_obs, 1))
            mixed_model = MixedLM(
                endog=data,
                exog=design_matrix,
                groups=subj_id,
                exog_re=exog_re_simple,
            )
            result = mixed_model.fit(reml=False, maxiter=500)

        beta = np.array(result.fe_params)
        rand_cov = np.array(result.cov_re)
        residuals = data - result.fittedvalues.values

        stats = ModelStats(
            bic=result.bic,
            logl=result.llf,
            sse=np.sum(residuals ** 2),
            covb=np.array(result.cov_params().iloc[:len(beta), :len(beta)]),
            sebeta=np.array(result.bse_fe),
            iwres=residuals,
        )

    return EstimatedModel(
        input=model_input,
        order=m_order,
        design_matrix=design_matrix,
        design_vars=design_vars,
        beta=beta,
        rand_cov=rand_cov,
        stats=stats,
    )
