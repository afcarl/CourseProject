import numpy as np
import matplotlib.pyplot as plt
import time

from gaussian_process import GaussianProcess, gp_plot_class_data, plot_performance_hyper_parameter, \
    plot_performance_errors
from covariance_functions import CovarianceFamily, SquaredExponential, GammaExponential, ScaledSquaredExponential

data_params = np.array([1.0, 0.25, 0.05])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'class')
num = 200
test_density = 50
dim = 2
seed = 21

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

model_params = np.array([2.2, np.exp(1.73), 0.2])
model_covariance_obj = ScaledSquaredExponential(model_params)
new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'class')
new_gp.find_hyper_parameters(x_tr, y_tr, max_iter=50, alternate=False)

print(new_gp.covariance_obj.get_params())
predicted_y_test = new_gp.predict(x_test, x_tr, y_tr)
mistake_lst = [i for i in range(len(y_test)) if predicted_y_test[i] != y_test[i]]

print("Mistakes: ", np.sum(predicted_y_test != y_test))
if dim == 2:
    gp_plot_class_data(x_tr, y_tr, 'bo', 'ro')
    plt.contour(np.linspace(0, 1, test_density), np.linspace(0, 1, test_density),
                y_test.reshape((test_density, test_density)), levels=[0], colors=['g'])
    plt.contour(np.linspace(0, 1, test_density), np.linspace(0, 1, test_density),
                predicted_y_test.reshape((test_density, test_density)), levels=[0], colors=['y'])
    plt.show()
