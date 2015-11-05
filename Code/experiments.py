import numpy as np
import matplotlib.pyplot as plt
import time

from gaussian_process import GaussianProcess, gp_plot_class_data, plot_performance_hyper_parameter, \
    plot_performance_errors
from covariance_functions import CovarianceFamily, SquaredExponential, GammaExponential, ScaledSquaredExponential, \
    SuperScaledSquaredExponential

data_params = np.array([1.0, 0.25, 0.05])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'class')
num = 100
test_density = 50
dim = 2
seed = 21
iterations = 300
plot_iterations = 100
first_algorithm_label = 'log-Scaled SE'
second_algorithm_label = 'Not log-Scaled SE'
third_algorithm_label = 'Half-log-scaled SE'

np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 2:
    x_1 = np.linspace(0, 1, test_density)
    x_2 = np.linspace(0, 1, test_density)
    x_1, x_2 = np.meshgrid(x_1, x_2)
    x_test = np.array(list(zip(x_1.reshape(-1).tolist(), x_2.reshape(-1).tolist())))
    x_test = x_test.T
else:
    x_test = np.random.rand(dim, test_density**2)

print("Generating Data")

y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)

print("Data generated")

# First method
model_params = np.array([np.exp(2.2), np.exp(1.73), np.exp(0.2)])
model_covariance_obj = SuperScaledSquaredExponential(model_params)
first_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'class')
w_a_list, time_a_list = first_gp.find_hyper_parameters(x_tr, y_tr, max_iter=iterations, alternate=False)
w_a_opt = first_gp.covariance_obj.get_params()
w_a_list = w_a_list[:plot_iterations]
time_a_list = time_a_list[:plot_iterations]

# Second method
model_params = np.array([2.2, 1.73, 0.2])
model_covariance_obj = SquaredExponential(model_params)
second_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'class')
w_list, time_list = second_gp.find_hyper_parameters(x_tr, y_tr, max_iter=iterations, alternate=False)
w_opt = second_gp.covariance_obj.get_params()
w_list = w_list[:plot_iterations]
time_list = time_list[:plot_iterations]

# # Third method
# model_params = np.array([2.2, np.exp(1.73), 0.2])
# model_covariance_obj = ScaledSquaredExponential(model_params)
# second_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'class')
# w_sa_list, time_sa_list = second_gp.find_hyper_parameters(x_tr, y_tr, max_iter=iterations, alternate=False)
# w_sa_opt = second_gp.covariance_obj.get_params()
# w_sa_list = w_sa_list[:plot_iterations]
# time_sa_list = time_sa_list[:plot_iterations]

# Errors, time
# plot_performance_errors(w_a_list, first_gp, x_test, y_test, x_tr, y_tr, 'b', first_algorithm_label,
#                         time_list=time_a_list)
# plot_performance_errors(w_list, second_gp, x_test, y_test, x_tr, y_tr, 'r', second_algorithm_label,
#                         time_list=time_list)
# plt.legend()
# plt.show()

# Errors, iterations
# plot_performance_errors(w_a_list, first_gp, x_test, y_test, x_tr, y_tr, 'b', first_algorithm_label)
# plot_performance_errors(w_list, second_gp, x_test, y_test, x_tr, y_tr, 'r', second_algorithm_label)
# plt.legend()
# plt.show()

# Hyper-parameters, time
# plot_performance_hyper_parameter(w_a_list, w_a_opt,'b', first_algorithm_label, time_list=time_a_list)
# plot_performance_hyper_parameter(w_list, w_opt, 'r', second_algorithm_label, time_list=time_list)
# plt.legend()
# plt.show()

# Hyper-parameters, iterations
plot_performance_hyper_parameter(w_a_list, w_opt, 'b', first_algorithm_label)
plot_performance_hyper_parameter(w_list, w_opt, 'r', second_algorithm_label)
# plot_performance_hyper_parameter(w_sa_list, w_sa_opt, 'g', third_algorithm_label)
plt.legend()
plt.show()

