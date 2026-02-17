"""
estimate_model.py - Port of estimateModel.m

Fits a single mixed-effect model of specified order using statsmodels.
All data flows through pandas DataFrames/Series instead of raw numpy matrices.
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
    covb: pd.DataFrame  # covariance of fixed effects (named rows/cols)
    sebeta: pd.Series  # standard errors of fixed effects (named)
    iwres: pd.Series  # individual weighted residuals (indexed by observation)


@dataclass
class ModelInput:
    """Input data stored with the model as a single DataFrame."""
    df: pd.DataFrame  # columns: 'subj_id', 'age', 'data', optional 'group_1', 'group_2', 'cov_1', ...
    group_cols: list  # names of grouping columns (e.g. ['group_1'] or ['group_1','group_2'])
    cov_cols: list  # names of covariate columns (e.g. ['cov_1'])


@dataclass
class EstimatedModel:
    """Output of estimate_model (mirrors MATLAB outModel struct)."""
    input: ModelInput
    order: int
    design_matrix: pd.DataFrame  # named columns matching design_vars
    beta: pd.Series  # fixed effects indexed by design variable names
    rand_cov: Optional[pd.DataFrame]
    stats: ModelStats
    m_name: str = ""
    group_effect: Optional[dict] = None
    inter_effect: Optional[dict] = None

    @property
    def design_vars(self):
        """Column names of the design matrix."""
        return list(self.design_matrix.columns)


def _build_input_df(subj_id, grouping, age, cov, data):
    """
    Build a unified pandas DataFrame from separate input arrays.

    This is a convenience for callers still passing separate arrays.
    Returns (DataFrame, group_col_names, cov_col_names).
    """
    df = pd.DataFrame({
        'subj_id': np.asarray(subj_id).flatten(),
        'age': np.asarray(age, dtype=float).flatten(),
        'data': np.asarray(data, dtype=float).flatten(),
    })

    group_cols = []
    if grouping is not None:
        grouping = np.asarray(grouping, dtype=float)
        if grouping.ndim == 1:
            grouping = grouping.reshape(-1, 1)
        if grouping.shape[1] > 0:
            for i in range(grouping.shape[1]):
                col = f'group_{i + 1}'
                df[col] = grouping[:, i]
                group_cols.append(col)

    cov_cols = []
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        if cov.shape[1] > 0:
            for i in range(cov.shape[1]):
                col = f'cov_{i + 1}'
                df[col] = cov[:, i]
                cov_cols.append(col)

    return df, group_cols, cov_cols


def estimate_model(subj_id, grouping, age, cov, data, opts):
    """
    Fit a mixed-effect model of given order.

    Parameters
    ----------
    subj_id : array-like
        Subject IDs (#obs x 1).
    grouping : array-like or None
        Grouping variable. None for 1 group, 1 column for 2 groups,
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
        Fitted model with parameters, statistics, and design matrix
        all stored as pandas DataFrames/Series.
    """
    # Build unified DataFrame from inputs
    input_df, group_cols, cov_cols = _build_input_df(subj_id, grouping, age, cov, data)

    # Defaults
    group_effect = opts.get('group_effect', True)
    inter_effect = opts.get('inter_effect', True)
    m_type = opts.get('m_type', 'intercept')
    m_order = opts['m_order']

    if not group_effect:
        inter_effect = False

    # Determine number of groups
    n_groups = len(group_cols) + 1 if group_cols else 1

    if n_groups == 2:
        unique_vals = sorted(input_df[group_cols[0]].unique())
        if unique_vals != [0.0, 1.0]:
            raise ValueError("Error in group specification: expected 0 and 1 values")

    if n_groups == 3:
        for gc in group_cols:
            unique_vals = sorted(input_df[gc].unique())
            if unique_vals != [0.0, 1.0]:
                raise ValueError("Error in group specification: expected 0 and 1 values")

    if n_groups > 3:
        raise ValueError("More than 3 groups found")

    # --- Build design matrix as a DataFrame ---
    design_df = pd.DataFrame({'intercept': 1.0}, index=input_df.index)

    if group_effect:
        if n_groups > 1:
            for gc in group_cols:
                design_df[gc] = input_df[gc].values
        else:
            print("Warning: only one group, no group effect will be taken into account")
            group_effect = False
            inter_effect = False

    age_vals = input_df['age'].values
    if m_order > 0:
        for io in range(1, m_order + 1):
            col_name = 'age' if io == 1 else f'age_{io}'
            design_df[col_name] = age_vals ** io

            if inter_effect:
                for gc in group_cols:
                    grp_vals = input_df[gc].values
                    if io == 1:
                        inter_name = f'age_x_{gc}'
                    else:
                        inter_name = f'age{io}_x_{gc}'
                    design_df[inter_name] = (age_vals ** io) * grp_vals

    # Add covariates
    for cc in cov_cols:
        design_df[cc] = input_df[cc].values

    # Store input
    model_input = ModelInput(
        df=input_df.copy(),
        group_cols=group_cols,
        cov_cols=cov_cols,
    )

    n_obs = len(input_df)
    design_matrix_np = design_df.values
    response = input_df['data'].values
    subjects = input_df['subj_id'].values

    # --- Fit model ---
    if m_type == 'glm':
        # OLS (no random effects)
        ols_model = sm.OLS(response, design_matrix_np)
        result = ols_model.fit()

        beta = pd.Series(result.params, index=design_df.columns, name='beta')
        rand_cov = None
        covb_df = pd.DataFrame(result.cov_params(),
                               index=design_df.columns, columns=design_df.columns)
        stats = ModelStats(
            bic=result.bic,
            logl=result.llf,
            sse=result.ssr,
            covb=covb_df,
            sebeta=pd.Series(result.bse, index=design_df.columns, name='se'),
            iwres=pd.Series(result.resid, index=input_df.index, name='residuals'),
        )
    else:
        # Mixed model — build random effects design matrix
        if m_type == 'slope' and m_order > 0:
            exog_re = np.column_stack([np.ones(n_obs), age_vals])
        else:
            exog_re = np.ones((n_obs, 1))

        mixed_model = MixedLM(
            endog=response,
            exog=design_matrix_np,
            groups=subjects,
            exog_re=exog_re,
        )

        try:
            result = mixed_model.fit(reml=False, maxiter=500)
        except Exception:
            # Fall back to random intercept only
            exog_re_simple = np.ones((n_obs, 1))
            mixed_model = MixedLM(
                endog=response,
                exog=design_matrix_np,
                groups=subjects,
                exog_re=exog_re_simple,
            )
            result = mixed_model.fit(reml=False, maxiter=500)

        fe_params = np.array(result.fe_params)
        beta = pd.Series(fe_params, index=design_df.columns, name='beta')
        rand_cov = pd.DataFrame(np.array(result.cov_re))
        residuals = response - np.asarray(result.fittedvalues)

        n_fe = len(fe_params)
        covb_full = np.asarray(result.cov_params())[:n_fe, :n_fe]
        covb_df = pd.DataFrame(covb_full,
                               index=design_df.columns, columns=design_df.columns)
        stats = ModelStats(
            bic=result.bic,
            logl=result.llf,
            sse=float(np.sum(residuals ** 2)),
            covb=covb_df,
            sebeta=pd.Series(np.array(result.bse_fe), index=design_df.columns, name='se'),
            iwres=pd.Series(residuals, index=input_df.index, name='residuals'),
        )

    return EstimatedModel(
        input=model_input,
        order=m_order,
        design_matrix=design_df,
        beta=beta,
        rand_cov=rand_cov,
        stats=stats,
    )
