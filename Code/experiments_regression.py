import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from gaussian_process import GaussianProcess
from plotting import gp_plot_class_data, plot_performance_hyper_parameter, \
    plot_performance_errors, plot_performance_function, plot_smse_vs_time, gp_plot_reg_data
from covariance_functions import CovarianceFamily, SquaredExponential, GammaExponential


def calculate_smse(predicted_y, true_y, train_y):
    return np.linalg.norm(predicted_y - true_y)/np.linalg.norm(np.mean(train_y) - true_y)

data_params = np.array([1.1, 0.1, 0.1])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'reg')
num = 1000
test_num = 700
dim = 2
seed = 21
iterations = 100
plot_iterations = 80
first_algorithm_label = 'exp-Scaled SE'
second_algorithm_label = 'SE'

np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 1:
    x_test = np.linspace(0, 1, test_num)
    x_test = x_test.reshape(1, test_num)
else:
    x_test = np.random.rand(dim, test_num)
y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)

# # Inducing_points
# ip_times, ip_smse = [], []
# start, finish = 0, 0
# #
# for m in [10, 15, 20]:
#     print('m:', m)
#     model_params = np.array([1., 0.2, 0.2])
#     model_covariance_obj = SquaredExponential(model_params)
#     new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
#     start = time.time()
#     inducing_points, mean, cov, _, _, _ = new_gp.reg_find_inducing_inputs(x_tr, y_tr, m, max_iter=30)
#     finish = time.time()
#     predicted_y_test, high, low = new_gp.reg_inducing_points_predict(inducing_points, mean, cov, x_test)
#     ip_times.append(finish - start)
#     ip_smse.append(calculate_smse(predicted_y_test, y_test, y_tr))
#
# plot_smse_vs_time(ip_times, ip_smse, label='Variarional Inducing Points')

# Inducing points means
ipm_times, ipm_smse = [], []
start, finish = 0, 0

for m in [10, 15, 20, 40, 50, 70]:
    print('m:', m)
    model_params = np.array([1., 0.2, 0.2])
    model_covariance_obj = SquaredExponential(model_params)
    new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
    start = time.time()
    inducing_points, mean, cov, _, _, _ = new_gp.means_reg_find_inducing_inputs(x_tr, y_tr, m, max_iter=30)
    finish = time.time()
    predicted_y_test, high, low = new_gp.reg_inducing_points_predict(inducing_points, mean, cov, x_test)
    ipm_times.append(finish - start)
    ipm_smse.append(calculate_smse(predicted_y_test, y_test, y_tr))

plot_smse_vs_time(ipm_times, ipm_smse, '-gx', label='K-Means Inducing Points')

# Subset of data
sod_times, sod_smse = [], []
start, finish = 0, 0

for m in [10, 50, 100, 200, 300, 500]:
    print('m:', m)
    model_params = np.array([1., 0.2, 0.2])
    model_covariance_obj = SquaredExponential(model_params)

    start = time.time()
    means = KMeans(n_clusters=m)
    means.fit(x_tr.T)
    inducing_points = means.cluster_centers_.T
    targets = []
    for i in range(inducing_points.shape[1]):
        nbrs = NearestNeighbors(n_neighbors=1).fit(x_tr.T)
        mean = inducing_points[:, i]
        if dim == 1:
            mean = mean[:, None]
        _, indices = nbrs.kneighbors(mean)
        targets.append(y_tr[indices][0,0,0])
    targets = np.array(targets)[:, None]
    # print(targets.shape)
    small_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
    _, _, fun_lst = small_gp.find_hyper_parameters(inducing_points, targets, max_iter=30)
    finish = time.time()
    predicted_y_test, high, low = small_gp.predict(x_test, x_tr, y_tr)
    sod_times.append(finish - start)
    sod_smse.append(calculate_smse(predicted_y_test, y_test, y_tr))

plot_smse_vs_time(sod_times, sod_smse, '-rs', label='K-Means Inducing Points')
plt.legend()
plt.show()

# # First method
# model_params = np.array([np.log(1.), np.log(0.73), np.log(0.4)])
# model_covariance_obj = ExpScaledSquaredExponential(model_params)
# first_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
#
# w_a_list, time_a_list, fun_a_list = first_gp.find_hyper_parameters(x_tr, y_tr, max_iter=iterations)
# w_a_list = [np.exp(w) for w in w_a_list]
#
# w_a_opt = w_a_list[-1]
# fun_a_list = fun_a_list[:plot_iterations]
# w_a_list = w_a_list[:plot_iterations]
# time_a_list = time_a_list[:plot_iterations]
#
# # Second method
# model_params = np.array([5.2, 0.73, 0.3])
# model_covariance_obj = SquaredExponential(model_params)
# second_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
# w_list, time_list, fun_list = second_gp.find_hyper_parameters(x_tr, y_tr, max_iter=iterations)
# w_opt = second_gp.covariance_obj.get_params()
#
# fun_list = fun_list[:plot_iterations]
# w_list = w_list[:plot_iterations]
# time_list = time_list[:plot_iterations]
#
# print(np.linalg.norm(w_opt - w_a_opt))
# print("%.10f" % w_opt[0], "%.10f" % w_opt[1], "%.10f" % w_opt[2])

# Errors, time
# plot_performance_errors(w_a_list, first_gp, x_test, y_test, x_tr, y_tr, 'b', first_algorithm_label,
#                         time_list=time_a_list)
# plot_performance_errors(w_list, second_gp, x_test, y_test, x_tr, y_tr, 'r', second_algorithm_label,
#                         time_list=time_list)
# plt.legend()
# plt.title("Regression, n = " + str(num) + ", d = " + str(dim))
# plt.show()

# Errors, iterations
# plot_performance_errors(w_a_list, first_gp, x_test, y_test, x_tr, y_tr, 'b', first_algorithm_label)
# plot_performance_errors(w_list, second_gp, x_test, y_test, x_tr, y_tr, 'r', second_algorithm_label)
# plt.legend()
# plt.title("Regression, n = " + str(num) + ", d = " + str(dim))
# plt.show()

# fun_opt = min([fun_list[-1], fun_a_list[-1]])
# #function, time
# plot_performance_function(fun_a_list, fun_opt,'b', first_algorithm_label, time_list=time_a_list)
# plot_performance_function(fun_list, fun_opt, 'r', second_algorithm_label, time_list=time_list)
# plt.legend()
# plt.title("Regression, n = " + str(num) + ", d = " + str(dim))
# plt.show()

# #function, iterations
# plot_performance_function(fun_a_list, fun_opt,'b', first_algorithm_label)
# plot_performance_function(fun_list, fun_opt, 'r', second_algorithm_label)
# plt.legend()
# plt.title("Regression, n = " + str(num) + ", d = " + str(dim))
# plt.show()

# #Hyper-parameters, time
# plot_performance_hyper_parameter(w_a_list, w_opt,'b', first_algorithm_label, time_list=time_a_list)
# plot_performance_hyper_parameter(w_list, w_opt, 'r', second_algorithm_label, time_list=time_list)
# plt.legend()
# plt.title("Regression, n = " + str(num) + ", d = " + str(dim))
# plt.show()
#
# # Hyper-parameters, iterations
# plot_performance_hyper_parameter(w_a_list, w_opt, 'b', first_algorithm_label)
# plot_performance_hyper_parameter(w_list, w_opt, 'r', second_algorithm_label)
# # plot_performance_hyper_parameter(w_sa_list, w_sa_opt, 'g', third_algorithm_label)
# plt.legend()
# plt.title("Regression, n = " + str(num) + ", d = " + str(dim))
# plt.show()plt.title("Regression, n = ", num, "d = ", dim)

