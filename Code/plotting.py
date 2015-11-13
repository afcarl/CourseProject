import matplotlib.pyplot as plt
import numpy as np


def gp_plot_reg_data(x, y, *args, **kwargs):
    """
    :param x: points array
    :param y: target values array
    :param color: color
    :return: figure
    """
    if (not isinstance(x, np.ndarray) or
            not isinstance(y, np.ndarray)):
        raise TypeError("The first two arguments must be numpy arrays")

    loc_x = x.reshape((x.size, ))
    loc_y = y.reshape((y.size, ))
    plt.plot(loc_x, loc_y, *args, **kwargs)


def gp_plot_class_data(x, y, color1, color2):
    """
    :param x: points array
    :param y: target values array
    :param color1: color for the first class
    :param color2: color for the second class
    :return: figure
    """
    loc_y = y.reshape((y.shape[0],))
    plt.plot(x[0, loc_y == 1], x[1, loc_y == 1], color1)
    plt.plot(x[0, loc_y == -1], x[1, loc_y == -1], color2)


def plot_performance_function(fun_list, f_opt, color, lbl, time_list=None):
    plt.ylabel(r"$\log(||f(w_n) - f(w^*)||)$")
    fun_list = [np.log(np.linalg.norm(value - f_opt)) for value in fun_list if value - f_opt > np.exp(-15)]
    if time_list is None:
        plt.xlabel("Iteration number")
        plt.plot(range(len(fun_list)), fun_list, color,
                 label=lbl)
    else:
        time_list = time_list[:len(fun_list)]
        plt.xlabel("Time (s)")
        plt.plot(time_list, fun_list, color, label=lbl)


def plot_performance_hyper_parameter(w_list, w_opt, color, lbl, time_list=None):
    """
    :param w_list: an iteration-wise list of hyper-parameters
    :param optimal_w: optimal value of hyper-parameters
    :return:
    """
    plt.ylabel(r"$\log(||w_n - w^*||)$")
    w_list = [np.log(np.linalg.norm(value - w_opt)) for value in w_list if np.linalg.norm(value - w_opt) > np.exp(-15)]
    if time_list is None:
        plt.xlabel("Iteration number")
        plt.plot(range(len(w_list)), w_list, color,
                 label=lbl)
    else:
        plt.xlabel("Time (s)")
        plt.plot(time_list[:len(w_list)], w_list, color, label=lbl)


def plot_performance_errors(w_list, gp, test_points, test_labels, train_points, train_labels, color, lbl,
                            time_list=None):
    """
    Plots the iteration-wise error
    :param w_list: an iteration-wise list of hyper-parameters
    :param test_points: test data points
    :param test_labels: labels at test data-points
    :return:
    """
    plt.ylabel("Number of errors on test set")
    error_list = []
    for w in w_list:
        gp.covariance_obj.set_params(w)
        predicted_labels = gp.predict(test_points, train_points, train_labels)
        error_list.append(np.sum(predicted_labels != test_labels) / test_labels.shape[0])
    if time_list is None:
        plt.plot(range(len(w_list)), error_list, color, label=lbl)
        plt.xlabel("Iteration number")
    else:
        plt.plot(time_list, error_list, color, label=lbl)
        plt.xlabel("Time (s)")