import numpy as np

from inducing_inputs_experiments import run_methods
from GP.gaussian_process_regression import GPR
from GP.covariance_functions import SquaredExponential, Matern, GammaExponential

data_params = np.array([1.0, 0.5, 0.1])
data_covariance_obj = SquaredExponential(data_params)

model_params = np.array([0.7, 0.3, 0.2])
model_covariance_obj = SquaredExponential(model_params)

gp = GPR(data_covariance_obj)
num = 500
test_num = 100
input_nums = [5, 10, 15, 20, 30, 50]
dim = 1
seed = 10

file_name = 'd1_n500'
title = 'Generated dataset, n = ' + str(num) + ', d = ' + str(dim)

# Generating data points
np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 1:
    x_test = np.linspace(0, 1, test_num)
    x_test = x_test.reshape(1, test_num)
else:
    x_test = np.random.rand(dim, test_num)

y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)
run_methods(x_tr, y_tr, x_test, y_test, model_params, input_nums, file_name, title, True)

