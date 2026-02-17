import numpy as np


def compute_residuals(data, covariates):
    # 1. Convert to NumPy arrays
    data = np.asarray(data, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    # 2. Check if covariates are "wide" and transpose if necessary
    # If rows (2) < columns (143), flip it to 143x2
    if covariates.shape[0] < covariates.shape[1]:
        covariates = covariates.T

    # 3. Center the covariates (column-wise)
    covariates = covariates - covariates.mean(axis=0)

    # 4. Create ones_col with the CORRECT number of rows (143)
    num_subjects = covariates.shape[0]
    ones_col = np.ones((num_subjects, 1))

    # 5. Now the rows match (143 == 143), so this will work!
    X = np.column_stack((covariates, ones_col))

    # ... rest of your regression math ...
    beta, _, _, _ = np.linalg.lstsq(X, data, rcond=None)
    residuals = data - (X @ beta)

    return residuals, beta
