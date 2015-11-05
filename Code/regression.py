import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation

from gaussian_process import GaussianProcess, gp_plot_reg_data, gp_plot_class_data
from covariance_functions import SquaredExponential, GammaExponential, ScaledSquaredExponential, Matern

data_params = np.array([2.0, 0.7, 0.01])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'reg')
num = 100
test_num = 300
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

model_params = np.array([1., 0.5, 0.4])
model_covariance_obj = SquaredExponential(model_params)
new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
new_gp.find_hyper_parameters(x_tr, y_tr)
print(new_gp.covariance_obj.get_params())
predicted_y_test = new_gp.predict(x_test, x_tr, y_tr)
# predicted_y_tr = new_gp.predict(x_tr, x_tr, y_tr)

if dim == 1:
    gp_plot_reg_data(x_tr, y_tr, 'yo')
    gp_plot_reg_data(x_test, predicted_y_test, 'b')
    gp_plot_reg_data(x_test, y_test, '-y')
#

# print(np.linalg.norm(y_test - predicted_y_test) / np.linalg.norm(predicted_y_test - np.mean(y_tr)))
# print(np.linalg.norm(y_tr - predicted_y_tr) / np.linalg.norm(predicted_y_tr - np.mean(y_tr)))

if dim == 1:
    plt.show()

