import matplotlib.pyplot as plt
import numpy as np
from old_version_gp import GaussianProcess

from GP.covariance_functions import SquaredExponential, ExpScaledSquaredExponential
from GP.plotting import plot_performance_hyper_parameter

data_params = np.array([1.0, 0.25, 0.05])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'class')
num = 30
test_density = 50
dim = 2
seed = 21
iterations_1 = 500
iterations_2 = 150
plot_iterations_1 = 490
plot_iterations_2 = 140
first_algorithm_label = 'Exp-Scaled SE (Exact Gradient)'
second_algorithm_label = 'SE (Exact Gradient)'
# third_algorithm_label = 'Half-log-scaled SE'

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
model_params = np.array([np.log(2.2), np.log(1.73), np.log(0.2)])
model_covariance_obj = ExpScaledSquaredExponential(model_params)

# model_params = np.array([2.2, 1.73, 0.2])
# model_covariance_obj = SquaredExponential(model_params)
first_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'class')
w_a_list, time_a_list = first_gp.find_hyper_parameters(x_tr, y_tr, max_iter=iterations_1, alternate=True)
w_a_list = [np.exp(w) for w in w_a_list]

w_a_opt = w_a_list[-1]
w_a_list = w_a_list[:plot_iterations_1]
time_a_list = time_a_list[:plot_iterations_1]

# Second method
model_params = np.array([2.2, 1.73, 0.2])
model_covariance_obj = SquaredExponential(model_params)
second_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'class')
w_list, time_list = second_gp.find_hyper_parameters(x_tr, y_tr, max_iter=iterations_2, alternate=True)
# w_opt = second_gp.covariance_obj.get_params()
w_opt = w_list[-1]
w_list = w_list[:plot_iterations_2]
time_list = time_list[:plot_iterations_2]

print("Difference between the optimums: ", np.linalg.norm(w_opt - w_a_opt))
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
# plt.show()

# Errors, iterations
# plot_performance_errors(w_a_list, first_gp, x_test, y_test, x_tr, y_tr, 'b', first_algorithm_label)
# plot_performance_errors(w_list, second_gp, x_test, y_test, x_tr, y_tr, 'r', second_algorithm_label)
# plt.legend()
# plt.show()

#Hyper-parameters, time
plot_performance_hyper_parameter(w_a_list, w_opt,'b', first_algorithm_label, time_list=time_a_list)
plot_performance_hyper_parameter(w_list, w_opt, 'r', second_algorithm_label, time_list=time_list)
plt.title("Classification, n = " + str(num) + ", d = " + str(dim))
plt.legend()
plt.show()

# Hyper-parameters, iterations
plot_performance_hyper_parameter(w_a_list, w_opt, 'b', first_algorithm_label)
plot_performance_hyper_parameter(w_list, w_opt, 'r', second_algorithm_label)
# plot_performance_hyper_parameter(w_sa_list, w_sa_opt, 'g', third_algorithm_label)
plt.title("Classification, n = " + str(num) + ", d = " + str(dim))
plt.legend()
plt.show()

