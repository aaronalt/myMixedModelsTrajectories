"""
Mixed Models Trajectories - Python port
========================================

A toolbox for analyzing longitudinal developmental data using
mixed-effects regression models.

Ported from MATLAB by: Original MATLAB code by Daniela Zoeller & Kadir Mutlu
(Medical Image Processing Lab, EPFL/UniGe)
"""

from .estimate_model import estimate_model, EstimatedModel
from .fit_opt_model import fit_opt_model
from .likelihood_ratio_test import likelihood_ratio_test
from .fdr_correct import fdr_correct
from .plotting import plot_model, plot_data_and_fit, plot_residuals
from .effect_size import group_calculation_effect, inter_calculation_effect
from .results import plot_models_and_save_results
