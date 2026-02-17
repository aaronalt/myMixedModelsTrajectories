"""
plotting.py - Port of plotModel.m, plotDataAndFit.m, plotResiduals.m, plotCI.m

Publication-quality visualization of mixed models with matplotlib.
All model data accessed through pandas DataFrames/Series.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def plot_data_and_fit(age, data=None, subjects=None, params=None,
                      plot_color='blue', age_sorted=None, ypred=None, delta=None,
                      ax=None):
    """
    Plot longitudinal data points and a fitted model curve.

    Parameters
    ----------
    age : array-like or pd.Series
        Observation time points (x axis).
    data : array-like, pd.Series, or None
        Response data (y axis). If None, no data points plotted.
    subjects : array-like, pd.Series, or None
        Subject IDs for connecting repeated measures.
    params : array-like or None
        Polynomial coefficients [intercept, age, age^2, ...].
    plot_color : color
        Matplotlib color specification.
    age_sorted, ypred, delta : array-like or None
        For plotting confidence intervals.
    ax : matplotlib Axes or None

    Returns
    -------
    handle : matplotlib artist or None
    """
    if ax is None:
        ax = plt.gca()

    # Ensure numpy for plotting
    age = np.asarray(age)
    handle = None

    if data is not None:
        data = np.asarray(data)
        handle = ax.plot(age, data, '.', markersize=12, color=plot_color, alpha=0.5)[0]

        if subjects is not None:
            subjects = np.asarray(subjects)
            for sid in np.unique(subjects):
                mask = subjects == sid
                sort_idx = np.argsort(age[mask])
                ax.plot(age[mask][sort_idx], data[mask][sort_idx], '-', color=plot_color, alpha=0.3, linewidth=0.8)

    if params is not None:
        params = np.asarray(params)
        age_vec = np.linspace(age.min(), age.max(), 200)
        poly_coeffs = np.flip(params)
        ax.plot(age_vec, np.polyval(poly_coeffs, age_vec), linewidth=3,
                color=np.array(plot_color) * 0.8 if isinstance(plot_color, np.ndarray) else plot_color)

        if ypred is not None and delta is not None and age_sorted is not None:
            a_s = np.asarray(age_sorted)
            y_s = np.asarray(ypred)
            d_s = np.asarray(delta)
            sort_idx = np.argsort(a_s)
            a_s, y_s, d_s = a_s[sort_idx], y_s[sort_idx], d_s[sort_idx]
            color = np.array(plot_color) * 0.8 if isinstance(plot_color, np.ndarray) else plot_color
            ax.fill_between(a_s, y_s - d_s, y_s + d_s, alpha=0.2, color=color)

    return handle


def _extract_group_params(model, pl_model, groups, n_cov):
    """Extract per-group polynomial parameters from the beta Series."""
    beta = pl_model.beta.values  # convert to positional array
    order = model.order
    params = {}

    if groups == 1:
        params[0] = beta[:order + 1]
        return params

    if groups == 2:
        if order == 0:
            params[0] = np.array([beta[0] + beta[1]])
            params[1] = np.array([beta[0]])
        elif order == 1:
            params[0] = np.array([beta[0] + beta[1], beta[2] + beta[3]])
            params[1] = np.array([beta[0], beta[2]])
        elif order == 2:
            params[0] = np.array([beta[0] + beta[1], beta[2] + beta[3], beta[4] + beta[5]])
            params[1] = np.array([beta[0], beta[2], beta[4]])
        elif order == 3:
            params[0] = np.array([beta[0] + beta[1], beta[2] + beta[3],
                                  beta[4] + beta[5], beta[6] + beta[7]])
            params[1] = np.array([beta[0], beta[2], beta[4], beta[6]])

    elif groups == 3:
        if order == 0:
            params[0] = np.array([beta[0] + beta[1]])
            params[1] = np.array([beta[0] + beta[2]])
            params[2] = np.array([beta[0]])
        elif order == 1:
            params[0] = np.array([beta[0] + beta[1], beta[3] + beta[4]])
            params[1] = np.array([beta[0] + beta[2], beta[3] + beta[5]])
            params[2] = np.array([beta[0], beta[3]])
        elif order == 2:
            params[0] = np.array([beta[0] + beta[1], beta[3] + beta[4], beta[6] + beta[7]])
            params[1] = np.array([beta[0] + beta[2], beta[3] + beta[5], beta[6] + beta[8]])
            params[2] = np.array([beta[0], beta[3], beta[6]])
        elif order == 3:
            params[0] = np.array([beta[0] + beta[1], beta[3] + beta[4],
                                  beta[6] + beta[7], beta[9] + beta[10]])
            params[1] = np.array([beta[0] + beta[2], beta[3] + beta[5],
                                  beta[6] + beta[8], beta[9] + beta[11]])
            params[2] = np.array([beta[0], beta[3], beta[6], beta[9]])

    return params


def _extract_no_interaction_params(model, pl_model, groups, n_cov):
    """Extract per-group parameters from a no-interaction reduced model."""
    beta = pl_model.beta.values
    order = model.order
    params = {}

    if groups == 2:
        if order == 1:
            params[0] = np.array([beta[0] + beta[1], beta[2]])
            params[1] = np.array([beta[0], beta[2]])
        elif order == 2:
            params[0] = np.array([beta[0] + beta[1], beta[2], beta[3]])
            params[1] = np.array([beta[0], beta[2], beta[3]])
        elif order == 3:
            params[0] = np.array([beta[0] + beta[1], beta[2], beta[3], beta[4]])
            params[1] = np.array([beta[0], beta[2], beta[3], beta[4]])

    elif groups == 3:
        if order == 1:
            params[0] = np.array([beta[0] + beta[1], beta[3]])
            params[1] = np.array([beta[0] + beta[2], beta[3]])
            params[2] = np.array([beta[0], beta[3]])
        elif order == 2:
            params[0] = np.array([beta[0] + beta[1], beta[3], beta[4]])
            params[1] = np.array([beta[0] + beta[2], beta[3], beta[4]])
            params[2] = np.array([beta[0], beta[3], beta[4]])
        elif order == 3:
            params[0] = np.array([beta[0] + beta[1], beta[3], beta[4], beta[5]])
            params[1] = np.array([beta[0] + beta[2], beta[3], beta[4], beta[5]])
            params[2] = np.array([beta[0], beta[3], beta[4], beta[5]])

    return params


def _compute_confidence_intervals(pl_model, n_cov):
    """
    Compute prediction confidence intervals from a model.

    Uses the design_matrix DataFrame and covb DataFrame from the model.
    """
    input_df = pl_model.input.df
    age = input_df['age'].values
    sort_idx = np.argsort(age)
    age_sorted = age[sort_idx]

    # Exclude covariate columns from CI computation
    cov_cols = pl_model.input.cov_cols
    non_cov_cols = [c for c in pl_model.design_matrix.columns if c not in cov_cols]

    beta = pl_model.beta[non_cov_cols].values
    des_mat = pl_model.design_matrix[non_cov_cols].values[sort_idx]
    cov_b = pl_model.stats.covb.loc[non_cov_cols, non_cov_cols].values
    residuals = pl_model.stats.iwres.values[sort_idx]

    ypred = des_mat @ beta

    n = len(age)
    p = len(beta)
    mse = np.sum(residuals ** 2) / max(n - p, 1)
    var_pred = np.sum((des_mat @ cov_b) * des_mat, axis=1)
    t_crit = stats.t.ppf(0.975, max(n - p, 1))
    delta = t_crit * np.sqrt(var_pred + mse)

    return age_sorted, ypred, delta, sort_idx


def _split_by_group(model):
    """
    Split model input data by group, returning dicts of per-group arrays
    and a boolean group_logic array.
    """
    input_df = model.input.df
    group_cols = model.input.group_cols
    n_groups = len(group_cols) + 1 if group_cols else 1
    n_obs = len(input_df)

    group_data = {}
    group_age = {}
    group_subj = {}
    group_logic = np.zeros((n_obs, n_groups), dtype=bool)

    if n_groups == 1:
        group_data[0] = input_df['data'].values
        group_age[0] = input_df['age'].values
        group_subj[0] = input_df['subj_id'].values
        group_logic[:, 0] = True
    elif n_groups == 2:
        mask1 = input_df[group_cols[0]].values.astype(bool)
        group_data[0] = input_df.loc[mask1, 'data'].values
        group_age[0] = input_df.loc[mask1, 'age'].values
        group_subj[0] = input_df.loc[mask1, 'subj_id'].values
        group_data[1] = input_df.loc[~mask1, 'data'].values
        group_age[1] = input_df.loc[~mask1, 'age'].values
        group_subj[1] = input_df.loc[~mask1, 'subj_id'].values
        group_logic[:, 0] = mask1
        group_logic[:, 1] = ~mask1
    elif n_groups == 3:
        for ig in range(2):
            mask = input_df[group_cols[ig]].values.astype(bool)
            group_data[ig] = input_df.loc[mask, 'data'].values
            group_age[ig] = input_df.loc[mask, 'age'].values
            group_subj[ig] = input_df.loc[mask, 'subj_id'].values
            group_logic[:, ig] = mask
        mask3 = (input_df[group_cols[0]].values == 0) & (input_df[group_cols[1]].values == 0)
        group_data[2] = input_df.loc[mask3, 'data'].values
        group_age[2] = input_df.loc[mask3, 'age'].values
        group_subj[2] = input_df.loc[mask3, 'subj_id'].values
        group_logic[:, 2] = mask3

    return n_groups, group_data, group_age, group_subj, group_logic


def plot_model(model, plot_opts, fig=None):
    """
    Plot a fitted mixed-effect model with data, curves, and CIs.

    Parameters
    ----------
    model : EstimatedModel
        Fitted model to plot.
    plot_opts : dict
        Plotting options:
        - 'plot_type': 'full', 'redInter', or 'redGrp'
        - 'plot_col': list of colors per group
        - 'plot_ci': bool, whether to plot confidence intervals
        - 'n_cov': number of covariates
        - 'leg_txt': list of legend labels
        - 'x_label', 'y_label': axis labels
        - 'title_text': optional custom title
        - 'fig_size': tuple (width, height) in inches
    fig : matplotlib Figure or None

    Returns
    -------
    fig : matplotlib Figure
    plotted_model : str
        'full', 'noGroup', or 'noInteraction'
    """
    plot_col = plot_opts.get('plot_col', [np.array([0, 0, 1.]),
                                          np.array([1., 0, 0]),
                                          np.array([0, 0, 0.])])
    plot_type = plot_opts.get('plot_type', 'redGrp')
    plot_ci = plot_opts.get('plot_ci', True)
    n_cov = plot_opts.get('n_cov', 0)
    fig_size = plot_opts.get('fig_size', (8, 5))

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        ax = fig.gca()

    groups, group_data, group_age, group_subj, group_logic = _split_by_group(model)
    handles = []

    # --- Case 1: redGrp and group effect not significant ---
    if (plot_type == 'redGrp'
            and (model.group_effect is None or not model.group_effect['h'])):

        pl_model = model.group_effect['reduced_model']
        cov_cols = pl_model.input.cov_cols
        non_cov_cols = [c for c in pl_model.beta.index if c not in cov_cols]
        params = pl_model.beta[non_cov_cols].values

        for ig in range(groups):
            h = plot_data_and_fit(group_age[ig], group_data[ig], group_subj[ig],
                                  None, plot_col[ig], ax=ax)
            handles.append(h)

        if plot_ci:
            age_s, ypred, delta, _ = _compute_confidence_intervals(pl_model, n_cov)
            plot_data_and_fit(model.input.df['age'].values, None, None, params, plot_col[-1],
                              age_sorted=age_s, ypred=ypred, delta=delta, ax=ax)
        else:
            plot_data_and_fit(model.input.df['age'].values, None, None, params, plot_col[-1], ax=ax)

        plotted_model = 'noGroup'

    # --- Case 2: no significant interaction ---
    elif (model.order > 0
          and plot_type != 'full'
          and model.inter_effect is not None
          and not model.inter_effect['h']):

        for ig in range(groups):
            h = plot_data_and_fit(group_age[ig], group_data[ig], group_subj[ig],
                                  None, plot_col[ig], ax=ax)
            handles.append(h)

        pl_model = model.inter_effect['reduced_model']
        grp_params = _extract_no_interaction_params(model, pl_model, groups, n_cov)

        if plot_ci:
            age_s, ypred, delta, sort_idx = _compute_confidence_intervals(pl_model, n_cov)
            for ig in range(groups):
                var_id = group_logic[:, ig]
                y_g = ypred[var_id[sort_idx]]
                d_g = delta[var_id[sort_idx]]
                a_g = age_s[var_id[sort_idx]]
                plot_data_and_fit(group_age[ig], None, None, grp_params[ig],
                                  plot_col[ig], age_sorted=a_g, ypred=y_g, delta=d_g, ax=ax)
        else:
            for ig in range(groups):
                plot_data_and_fit(group_age[ig], None, None, grp_params[ig],
                                  plot_col[ig], ax=ax)

        plotted_model = 'noInteraction'

    # --- Case 3: full model ---
    else:
        for ig in range(groups):
            h = plot_data_and_fit(group_age[ig], group_data[ig], group_subj[ig],
                                  None, plot_col[ig], ax=ax)
            handles.append(h)

        pl_model = model

        if groups > 1:
            grp_params = _extract_group_params(model, pl_model, groups, n_cov)
        else:
            grp_params = {0: pl_model.beta.values[:model.order + 1]}

        if plot_ci:
            age_s, ypred, delta, sort_idx = _compute_confidence_intervals(pl_model, n_cov)
            for ig in range(groups):
                var_id = group_logic[:, ig]
                y_g = ypred[var_id[sort_idx]]
                d_g = delta[var_id[sort_idx]]
                a_g = age_s[var_id[sort_idx]]
                plot_data_and_fit(group_age[ig], None, None, grp_params[ig],
                                  plot_col[ig], age_sorted=a_g, ypred=y_g, delta=d_g, ax=ax)
        else:
            for ig in range(groups):
                plot_data_and_fit(group_age[ig], None, None, grp_params[ig],
                                  plot_col[ig], ax=ax)

        plotted_model = 'full'

    # --- Labels and title ---
    title_text = plot_opts.get('title_text')
    if title_text:
        ax.set_title(title_text)
    else:
        if model.group_effect is None:
            ax.set_title(f"model order: {model.order}")
        elif model.order > 0 and model.inter_effect is not None:
            ax.set_title(
                f"model order: {model.order},   "
                f"p-val group effect: {model.group_effect['p']:.4f},   "
                f"p-val interaction: {model.inter_effect['p']:.4f}"
            )
        else:
            ax.set_title(
                f"model order: {model.order},   "
                f"p-val group effect: {model.group_effect['p']:.4f}"
            )

    ax.set_xlabel(plot_opts.get('x_label', 'age'))
    ax.set_ylabel(plot_opts.get('y_label', 'data'))

    leg_txt = plot_opts.get('leg_txt')
    valid_handles = [h for h in handles if h is not None]
    if leg_txt and valid_handles:
        ax.legend(valid_handles, leg_txt[:len(valid_handles)])
    elif valid_handles:
        leg_labels = [f'Group {ig + 1}' for ig in range(len(valid_handles))]
        ax.legend(valid_handles, leg_labels)

    return fig, plotted_model


def plot_residuals(model, fig=None):
    """
    Create a residual normal probability plot (QQ plot).

    Parameters
    ----------
    model : EstimatedModel
    fig : matplotlib Figure or None

    Returns
    -------
    fig : matplotlib Figure
    residuals : pd.Series
    """
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    else:
        ax = fig.gca()

    # Compute residuals: predicted - observed
    pred = model.design_matrix.values @ model.beta.values
    observed = model.input.df['data'].values
    residuals = pd.Series(pred - observed, name='residuals')

    stats.probplot(residuals.values, dist="norm", plot=ax)
    ax.set_title('Normal Probability Plot of Residuals')

    return fig, residuals
