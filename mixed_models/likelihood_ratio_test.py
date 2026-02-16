"""
likelihood_ratio_test.py - Port of likelihoodratiotest.m

Performs likelihood ratio tests to compare nested mixed models.
"""

from scipy.stats import chi2


def likelihood_ratio_test(logl_model, logl_null, dof, alpha=0.05):
    """
    Likelihood ratio test comparing a full model to a reduced (null) model.

    Parameters
    ----------
    logl_model : float
        Log-likelihood of the full model.
    logl_null : float
        Log-likelihood of the reduced model.
    dof : int
        Degrees of freedom (difference in number of parameters).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    h : bool
        True if null hypothesis rejected (full model significantly better).
    p : float
        p-value of the test statistic.
    dof : int
        Degrees of freedom used.
    lr_stat : float
        Likelihood ratio test statistic.
    """
    lr_stat = 2 * (logl_model - logl_null)
    p = 1 - chi2.cdf(lr_stat, dof)
    h = p <= alpha
    return h, p, dof, lr_stat
