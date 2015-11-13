import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation

from gaussian_process import GaussianProcess
from plotting import gp_plot_reg_data, gp_plot_class_data
from covariance_functions import SquaredExponential, GammaExponential, Matern, \
    ExpScaledSquaredExponential

data_params = np.array([2.0, 0.3, 0.01])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'reg')
num = 200
test_num = 500
dim = 1
seed = 21

np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 1:
    x_test = np.linspace(0, 1, test_num)
    x_test = x_test.reshape(1, test_num)
else:
    x_test = np.random.rand(dim, test_num)
y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)


model_params = np.array([1., 0.5, 0.2])
model_covariance_obj = SquaredExponential(model_params)

new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
# new_gp.find_hyper_parameters(x_tr, y_tr, max_iter=30)
# predicted_y_test, high, low = new_gp.predict(x_test, x_tr, y_tr)
inducing_points, mean, cov, _, _, _ = new_gp.reg_find_inducing_inputs(x_tr, y_tr, 7, max_iter=30)
predicted_y_test, high, low = new_gp.reg_inducing_points_predict(inducing_points, mean, cov, x_test)
print(new_gp.covariance_obj.get_params())
print(np.linalg.norm(predicted_y_test - y_test)/y_test.size)

if dim == 1:
    gp_plot_reg_data(x_tr, y_tr, 'yo')
    gp_plot_reg_data(x_test, predicted_y_test, 'b')
    gp_plot_reg_data(inducing_points, mean, 'ro', markersize=12)
    gp_plot_reg_data(x_test, low, 'g-')
    gp_plot_reg_data(x_test, high, 'r-')
    gp_plot_reg_data(x_test, y_test, 'y-')

if dim == 1:
    plt.show()

