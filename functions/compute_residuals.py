import numpy as np


def compute_residuals(data, covariates):
    num_of_subjects, num_of_regions = data.shape
    covariates = covariates - covariates.mean(axis=0)
    ones_col = np.ones((num_of_subjects, 1))
    X = np.column_stack((covariates, ones_col))
    beta, _, _, _ = np.linalg.lstsq(X, data, rcond=None)
    residuals = data - (X @ beta)

    return residuals, beta

