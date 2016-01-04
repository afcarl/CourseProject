import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from gaussian_process_regression import GPR
from plotting import gp_plot_reg_data, gp_plot_class_data
from covariance_functions import SquaredExponential, GammaExponential, Matern

data_params = np.array([1.1, 0.3, 0.1])
data_covariance_obj = SquaredExponential(data_params)
model_params = np.array([1., 0.1, 0.1])
model_covariance_obj = SquaredExponential(model_params)
gp = GPR(data_covariance_obj)
num = 100
test_num = 100
dim = 1
seed = 21
method = 'vi'  # possible methods: 'brute', 'vi', 'means'(, 'svi')
ind_inputs_num = 5

# Generating data points
np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 1:
    x_test = np.linspace(0, 1, test_num)
    x_test = x_test.reshape(1, test_num)
else:
    x_test = np.random.rand(dim, test_num)
y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)

if method == 'brute':
    new_gp = GPR(model_covariance_obj)
    new_gp.find_hyper_parameters(x_tr, y_tr, max_iter=30)
    predicted_y_test, high, low = new_gp.predict(x_test, x_tr, y_tr)

else:
    model_covariance_obj = SquaredExponential(model_params)
    new_gp = GPR(model_covariance_obj, method=method)
    inducing_points, mean, cov, _, _, _ = new_gp.find_inducing_inputs(x_tr, y_tr, ind_inputs_num, max_iter=30)
    predicted_y_test, high, low = new_gp.predict(inducing_points, mean, cov, x_test)


print(np.linalg.norm(predicted_y_test - y_test)/y_test.size)

if dim == 1:
    gp_plot_reg_data(x_tr, y_tr, 'yo')
    gp_plot_reg_data(x_test, predicted_y_test, 'b')
    gp_plot_reg_data(x_test, low, 'g-')
    gp_plot_reg_data(x_test, high, 'r-')
    gp_plot_reg_data(x_test, y_test, 'y-')
    if method != 'brute':
        gp_plot_reg_data(inducing_points, mean, 'ro', markersize=12)
    # gp_plot_reg_data(inducing_points, targets, 'ro', markersize=12)
    plt.show()