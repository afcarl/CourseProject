import numpy as np
from matplotlib import pyplot as plt
from matplotlib2tikz import save

from GP.covariance_functions import SquaredExponential
from GP.gaussian_process_regression import GPR

from sklearn.cluster import KMeans


def run_methods(train_points, train_targets, test_points, test_targets,
                model_parameters, optimizer_options, file_name, ind_num, title, metric_name='r2', show=False):

    means = KMeans(n_clusters=ind_num)
    means.fit(train_points.T)
    inputs = means.cluster_centers_.T

    method = 'svi'
    parametrization = 'cholesky'
    optimizer = 'L-BFGS-B'
    color = '-go'
    opts = optimizer_options[0]
    print('svi')
    model_covariance_obj = SquaredExponential(np.copy(model_parameters))
    new_gp = GPR(model_covariance_obj, method=method, parametrization=parametrization, optimizer=optimizer)
    res = new_gp.fit(train_points, train_targets, num_inputs=ind_num, optimizer_options=opts, inputs=inputs)
    name = 'svi-' + optimizer
    # print(res.params)
    if metric_name == 'r2':
        metric = lambda w: new_gp.get_prediction_quality(w, test_points, test_targets)
    # elif metric_name == 'loss':
    #     metric = lambda w: new_gp.get_loss(w, train_points, train_targets)
    else:
        raise ValueError('Unknown metric', metric_name)
    x_lst, y_lst = res.plot_performance(metric, 'i', freq=1)
    plt.plot(x_lst, y_lst, color, label=name)

    print('vi-means')
    method = 'means'
    opt_options = optimizer_options[1]

    model_covariance_obj = SquaredExponential(np.copy(model_parameters))
    new_gp = GPR(model_covariance_obj, method=method)
    res = new_gp.fit(train_points, train_targets, num_inputs=ind_num, optimizer_options=opt_options, inputs=inputs)
    name = 'vi-means'
    if metric_name == 'r2':
        metric = lambda w: new_gp.get_prediction_quality(w, train_points, train_targets, test_points, test_targets)
    # elif metric_name == 'loss':
    #     metric = lambda w: new_gp.get_loss(w, train_points, train_targets)
    else:
        raise ValueError('Unknown metric')
    x_lst, y_lst = res.plot_performance(metric, 'i', freq=1)
    plt.plot(x_lst, y_lst, '-kx', label=name)

    plt.xlabel('Epoch')
    if metric == 'r2':
        plt.ylabel('$R^2$-score on test data')
    elif metric_name == 'loss':
        plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    save('../Plots/vi_vs_svi/'+file_name)
    if show:
        plt.show()