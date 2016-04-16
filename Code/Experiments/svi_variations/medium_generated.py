import numpy as np

from experiments_svi_variations import run_methods
from GP.covariance_functions import SquaredExponential
from GP.gaussian_process_regression import GPR


data_params = np.array([1.1, 0.3, 0.1])
data_covariance_obj = SquaredExponential(data_params)
model_params = np.array([5.0, 1.0, 1.0])
model_covariance_obj = SquaredExponential(model_params)
gp = GPR(data_covariance_obj)
num = 4000
test_num = 500
dim = 5
seed = 21
ind_inputs_num = 200
max_iter = 300
batch_size = 200
title='generated dataset, n = 4000, d = 5, m=200'
file_name = 'medium_generated'

# Generating data points
np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 1:
    x_test = np.linspace(0, 1, test_num)
    x_test = x_test.reshape(1, test_num)
else:
    x_test = np.random.rand(dim, test_num)
y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)

# Cholesky parametrization

sag_options = {'maxiter':max_iter, 'batch_size': batch_size, 'print_freq': 10}
fg_options = {'maxiter':max_iter, 'print_freq': 10}
lbfgsb_options = {'maxiter': max_iter, 'disp': False}
sg_options = {'maxiter':max_iter, 'batch_size': batch_size, 'print_freq': 10, 'step0': 1e-5, 'gamma': 0.9}

optimizer_options = [sag_options, fg_options, lbfgsb_options, sg_options]

run_methods(x_tr, y_tr, x_test, y_test, model_params, optimizer_options, file_name, ind_inputs_num, title, False)