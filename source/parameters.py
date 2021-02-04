"""
Module containing the default parameters to use.
"""

regression_params = {

    # Dataset parameters.
    'data_cap': -1,

    # Iterations.
    'optimization_iterations': 10,  # If not zero, will re-optimize targets after the number of steps specified.
    'lambda_reg': 5,
    'postprocessing_step': False,  # If True, there will be a final optimization of ANN weigths to match constraints.

    # Cross validation.
    'n_fold': 5,

    # LR params
    'lr_lambda': 5,
    'lr_gamma': 0,

    # Constraints.
    'constraint_tolerance': 1e-2,
    'constraint_pvalue': 1e-1,  # percentage value of the original dataset discrimination value.
    'random_opt_set_size': 1,
}

classification_params = {

    # Dataset parameters.
    'data_cap': -1,

    # Iteration.
    'optimization_iterations': 10,  # If not zero, will re-optimize targets after the number of steps specified.
    'lambda_reg': 10,
    'postprocessing_step': False,  # If True, there will be a final optimization of ANN weigths to match constraints.
    'n_fold': 5,

    # Constraints.
    'constraint_tolerance': 1e-2,
    'constraint_pvalue': .5,  # percentage value of the original dataset discrimination value.
    'random_opt_set_size': 1,

    # Plots.
    'plot_skip_points': 5,
}
