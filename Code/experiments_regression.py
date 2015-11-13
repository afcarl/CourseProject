import numpy as np
import matplotlib.pyplot as plt
import time

from gaussian_process import GaussianProcess
from plotting import gp_plot_class_data, plot_performance_hyper_parameter, \
    plot_performance_errors, plot_performance_function
from covariance_functions import CovarianceFamily, SquaredExponential, GammaExponential, ExpScaledSquaredExponential

data_params = np.array([2.0, 0.7, 0.01])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'reg')
num = 500
test_num = 300
dim = 5
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

model_params = np.array([1., 0.5, 0.4])
model_covariance_obj = SquaredExponential(model_params)
new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
# new_gp.find_hyper_parameters(x_tr, y_tr)
# print(new_gp.covariance_obj.get_params())
# predicted_y_test = new_gp.predict(x_test, x_tr, y_tr)


# First method
model_params = np.array([np.log(1.), np.log(0.73), np.log(0.4)])
model_covariance_obj = ExpScaledSquaredExponential(model_params)
first_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')

########################################
first_gp.reg_plot_marginal_likelihood(x_tr, y_tr)
exit(0)
########################################
w_a_list, time_a_list, fun_a_list = first_gp.find_hyper_parameters(x_tr, y_tr, max_iter=iterations)
w_a_list = [np.exp(w) for w in w_a_list]

w_a_opt = w_a_list[-1]
fun_a_list = fun_a_list[:plot_iterations]
w_a_list = w_a_list[:plot_iterations]
time_a_list = time_a_list[:plot_iterations]

# Second method
model_params = np.array([5.2, 0.73, 0.3])
model_covariance_obj = SquaredExponential(model_params)
second_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
w_list, time_list, fun_list = second_gp.find_hyper_parameters(x_tr, y_tr, max_iter=iterations)
w_opt = second_gp.covariance_obj.get_params()

fun_list = fun_list[:plot_iterations]
w_list = w_list[:plot_iterations]
time_list = time_list[:plot_iterations]

print(np.linalg.norm(w_opt - w_a_opt))
print("%.10f" % w_opt[0], "%.10f" % w_opt[1], "%.10f" % w_opt[2])

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
# plt.title("Regression, n = " + str(num) + ", d = " + str(dim))
# plt.show()

# Errors, iterations
# plot_performance_errors(w_a_list, first_gp, x_test, y_test, x_tr, y_tr, 'b', first_algorithm_label)
# plot_performance_errors(w_list, second_gp, x_test, y_test, x_tr, y_tr, 'r', second_algorithm_label)
# plt.legend()
# plt.title("Regression, n = " + str(num) + ", d = " + str(dim))
# plt.show()

fun_opt = min([fun_list[-1], fun_a_list[-1]])
#function, time
plot_performance_function(fun_a_list, fun_opt,'b', first_algorithm_label, time_list=time_a_list)
plot_performance_function(fun_list, fun_opt, 'r', second_algorithm_label, time_list=time_list)
plt.legend()
plt.title("Regression, n = " + str(num) + ", d = " + str(dim))
plt.show()

#function, iterations
plot_performance_function(fun_a_list, fun_opt,'b', first_algorithm_label)
plot_performance_function(fun_list, fun_opt, 'r', second_algorithm_label)
plt.legend()
plt.title("Regression, n = " + str(num) + ", d = " + str(dim))
plt.show()

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

