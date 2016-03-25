import numpy as np
from matplotlib import pyplot as plt
from matplotlib2tikz import save

from GP.covariance_functions import SquaredExponential
from GP.gaussian_process_regression import GPR


def run_methods(train_points, train_targets, test_points, test_targets,
                model_parameters, optimizer_options, file_name, ind_num, title, show=False):

    method = 'svi'
    # parametrization = 'cholesky'
    # for optimizer, color, opts in zip(['SAG', 'FG', 'L-BFGS-B'], ['-ro', '-bo', '-go'],
    #                                   optimizer_options[:-1]):
    #     print('Optimizer', optimizer)
    #     model_covariance_obj = SquaredExponential(np.copy(model_parameters))
    #     new_gp = GPR(model_covariance_obj, method=method, parametrization=parametrization, optimizer=optimizer)
    #     res = new_gp.fit(train_points, train_targets, num_inputs=ind_num, optimizer_options=opts)
    #     name = 'svi-' + optimizer
    #     metric = lambda w: new_gp.get_prediction_quality(w, test_points, test_targets)
    #     x_lst, y_lst = res.plot_performance(metric, 'i', freq=10)
    #     plt.plot(x_lst, y_lst, color, label=name)

    parametrization = 'natural'
    print('Natural parametrization')

    opt_options = optimizer_options[-1]

    model_covariance_obj = SquaredExponential(np.copy(model_parameters))
    new_gp = GPR(model_covariance_obj, method=method, parametrization=parametrization)
    res = new_gp.fit(train_points, train_targets, num_inputs=ind_num, optimizer_options=opt_options)
    name = 'svi-natural'
    metric = lambda w: new_gp.get_prediction_quality(w, test_points, test_targets)
    x_lst, y_lst = res.plot_performance(metric, 'i', freq=10)
    plt.plot(x_lst, y_lst, '-yo', label=name)

    plt.xlabel('Epoch')
    plt.ylabel('MSE on test data')
    plt.legend()
    plt.title(title)
    save('../Plots/svi_variations/'+file_name)
    if show:
        plt.show()