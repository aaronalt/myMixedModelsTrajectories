"""
plotting.py - Port of plotModel.m, plotDataAndFit.m, plotResiduals.m, plotCI.m

Publication-quality visualization of mixed models with matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_data_and_fit(age, data=None, subjects=None, params=None,
                      plot_color='blue', age_sorted=None, ypred=None, delta=None,
                      ax=None):
    """
    Plot longitudinal data points and a fitted model curve.

    Port of plotDataAndFit.m.

    Parameters
    ----------
    age : array-like
        Observation time points (x axis).
    data : array-like or None
        Response data (y axis). If None, no data points plotted.
    subjects : array-like or None
        Subject IDs for connecting repeated measures.
    params : array-like or None
        Polynomial coefficients [intercept, age, age^2, ...].
        If None, no curve is plotted.
    plot_color : color
        Matplotlib color specification.
    age_sorted : array-like or None
        Sorted age for CI plotting.
    ypred : array-like or None
        Predicted values for CI.
    delta : array-like or None
        CI half-widths.
    ax : matplotlib Axes or None
        Axes to plot on. If None, uses current axes.

    Returns
    -------
    handle : matplotlib artist
        Plot handle for legend.
    """
    if ax is None:
        ax = plt.gca()

    handle = None

    # Plot data points
    if data is not None:
        handle = ax.plot(age, data, '.', markersize=12, color=plot_color, alpha=0.5)[0]

        # Connect repeated measurements per subject
        if subjects is not None:
            unique_sub = np.unique(subjects)
            for sid in unique_sub:
                mask = subjects == sid
                ax.plot(age[mask], data[mask], '--', color=plot_color, alpha=0.5, linewidth=0.8)

    # Plot fitted polynomial curve
    if params is not None:
        age_vec = np.linspace(np.min(age), np.max(age), 200)
        # params = [intercept, age, age^2, ...] -> polyval needs [highest, ..., lowest]
        poly_coeffs = np.flip(params)
        ax.plot(age_vec, np.polyval(poly_coeffs, age_vec), linewidth=3,
                color=np.array(plot_color) * 0.8 if isinstance(plot_color, np.ndarray) else plot_color)

        # Plot confidence intervals (shaded)
        if ypred is not None and delta is not None and age_sorted is not None:
            sort_idx = np.argsort(age_sorted)
            a_s = np.array(age_sorted)[sort_idx]
            y_s = np.array(ypred)[sort_idx]
            d_s = np.array(delta)[sort_idx]
            color = np.array(plot_color) * 0.8 if isinstance(plot_color, np.ndarray) else plot_color
            ax.fill_between(a_s, y_s - d_s, y_s + d_s, alpha=0.2, color=color)

    return handle


def _extract_group_params(model, pl_model, groups, n_cov):
    """
    Extract per-group polynomial parameters from beta vector.

    Mirrors the parameter extraction logic in plotModel.m.
    """
    beta = pl_model.beta
    order = model.order
    params = {}

    if groups == 1:
        params[0] = beta[:order + 1] if n_cov == 0 else beta[:order + 1]
        return params

    if groups == 2:
        if order == 0:
            params[0] = np.array([beta[0] + beta[1]])  # group 1
            params[1] = np.array([beta[0]])  # group 0 (reference)
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
    """
    Extract per-group parameters from a no-interaction reduced model.

    In the no-interaction model, the age terms are shared across groups,
    only intercepts differ.
    """
    beta = pl_model.beta
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


def _compute_confidence_intervals(pl_model, age, n_cov):
    """
    Compute prediction confidence intervals.

    Mimics MATLAB's nlpredci for linear models.
    """
    sort_idx = np.argsort(age)
    age_sorted = age[sort_idx]
    beta = pl_model.beta[:len(pl_model.beta) - n_cov] if n_cov > 0 else pl_model.beta
    des_mat = pl_model.design_matrix[sort_idx, :len(beta)]
    cov_b = pl_model.stats.covb[:len(beta), :len(beta)]
    residuals = pl_model.stats.iwres[sort_idx]

    # Predicted values
    ypred = des_mat @ beta

    # Standard error of prediction
    # delta = t_crit * sqrt(diag(X * CovB * X'))
    n = len(age)
    p = len(beta)
    mse = np.sum(residuals ** 2) / max(n - p, 1)
    var_pred = np.sum((des_mat @ cov_b) * des_mat, axis=1)
    t_crit = stats.t.ppf(0.975, max(n - p, 1))
    delta = t_crit * np.sqrt(var_pred + mse)

    return age_sorted, ypred, delta, sort_idx


def plot_model(model, plot_opts, fig=None):
    """
    Plot a fitted mixed-effect model with data, curves, and CIs.

    Port of plotModel.m.

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
        If None, creates a new figure.

    Returns
    -------
    fig : matplotlib Figure
    plotted_model : str
        'full', 'noGroup', or 'noInteraction'
    """
    # Defaults
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

    # Determine groups and split data
    grouping = model.input.grouping
    if grouping.shape[1] == 0:
        groups = 1
    else:
        groups = grouping.shape[1] + 1

    age_all = model.input.age
    data_all = model.input.data
    subj_all = model.input.subj_id

    # Split data by group
    group_data = {}
    group_age = {}
    group_subj = {}
    group_logic = np.zeros((len(age_all), groups), dtype=bool)

    if groups == 1:
        group_data[0] = data_all
        group_age[0] = age_all
        group_subj[0] = subj_all
        group_logic[:, 0] = True
    elif groups == 2:
        mask1 = grouping[:, 0].astype(bool)
        group_data[0] = data_all[mask1]
        group_age[0] = age_all[mask1]
        group_subj[0] = subj_all[mask1]
        group_data[1] = data_all[~mask1]
        group_age[1] = age_all[~mask1]
        group_subj[1] = subj_all[~mask1]
        group_logic[:, 0] = mask1
        group_logic[:, 1] = ~mask1
    elif groups == 3:
        for ig in range(2):
            mask = grouping[:, ig].astype(bool)
            group_data[ig] = data_all[mask]
            group_age[ig] = age_all[mask]
            group_subj[ig] = subj_all[mask]
            group_logic[:, ig] = mask
        # Third group: neither in group 1 nor group 2
        mask3 = (grouping[:, 0] == 0) & (grouping[:, 1] == 0)
        group_data[2] = data_all[mask3]
        group_age[2] = age_all[mask3]
        group_subj[2] = subj_all[mask3]
        group_logic[:, 2] = mask3

    handles = []

    # --- Decide which model to plot ---
    # Case 1: redGrp and group effect not significant -> plot no-group model
    if (plot_type == 'redGrp'
            and (model.group_effect is None or not model.group_effect['h'])):

        pl_model = model.group_effect['reduced_model']
        params = pl_model.beta[:len(pl_model.beta) - n_cov]

        # Plot data for each group
        for ig in range(groups):
            h = plot_data_and_fit(group_age[ig], group_data[ig], group_subj[ig],
                                  None, plot_col[ig], ax=ax)
            handles.append(h)

        # Plot single fitted line
        if plot_ci:
            age_s, ypred, delta, _ = _compute_confidence_intervals(pl_model, age_all, n_cov)
            plot_data_and_fit(age_all, None, None, params, plot_col[-1],
                              age_sorted=age_s, ypred=ypred, delta=delta, ax=ax)
        else:
            plot_data_and_fit(age_all, None, None, params, plot_col[-1], ax=ax)

        plotted_model = 'noGroup'

    # Case 2: no significant interaction -> plot no-interaction model
    elif (model.order > 0
          and plot_type != 'full'
          and model.inter_effect is not None
          and not model.inter_effect['h']):

        # Plot data
        for ig in range(groups):
            h = plot_data_and_fit(group_age[ig], group_data[ig], group_subj[ig],
                                  None, plot_col[ig], ax=ax)
            handles.append(h)

        pl_model = model.inter_effect['reduced_model']
        grp_params = _extract_no_interaction_params(model, pl_model, groups, n_cov)

        if plot_ci:
            age_s, ypred, delta, sort_idx = _compute_confidence_intervals(pl_model, age_all, n_cov)
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

    # Case 3: full model
    else:
        # Plot data
        for ig in range(groups):
            h = plot_data_and_fit(group_age[ig], group_data[ig], group_subj[ig],
                                  None, plot_col[ig], ax=ax)
            handles.append(h)

        pl_model = model

        if groups > 1:
            grp_params = _extract_group_params(model, pl_model, groups, n_cov)
        else:
            grp_params = {0: pl_model.beta[:model.order + 1]}

        if plot_ci:
            age_s, ypred, delta, sort_idx = _compute_confidence_intervals(pl_model, age_all, n_cov)
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

    # Legend
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
    Create a residual normal probability plot.

    Port of plotResiduals.m.

    Parameters
    ----------
    model : EstimatedModel
        Fitted model.
    fig : matplotlib Figure or None
        If None, creates a new figure.

    Returns
    -------
    fig : matplotlib Figure
    residuals : np.ndarray
    """
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    else:
        ax = fig.gca()

    # Compute residuals: predicted - observed
    pred = model.design_matrix @ model.beta
    residuals = pred - model.input.data

    # Normal probability plot (QQ plot)
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Normal Probability Plot of Residuals')

    return fig, residuals
