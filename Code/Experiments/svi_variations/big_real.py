import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file

from experiments_svi_variations import run_methods
from GP.covariance_functions import SquaredExponential

file_name = 'big_real.tikz'
model_params = np.array([3.0, 2.0, 1.0])
model_covariance_obj = SquaredExponential(model_params)
ind_inputs_num = 200
max_iter = 100
batch_size = 500

x_tr, y_tr = load_svmlight_file('../../../../Programming/DataSets/Regression/cadata(20640, 8).txt')
data_name = 'cadata'
title='cadata dataset, n = 10000, d = 8, m=200'

x_tr = x_tr.T
x_tr = x_tr.toarray()
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)


x_tr = (x_tr + 1) / 2
y_tr = y_tr.reshape((y_tr.size, 1))
x_test = x_tr[:, 10000:11000]
y_test = y_tr[10000:11000, :]
y_tr = y_tr[:10000, :]
x_tr = x_tr[:, :10000]


# Cholesky parametrization

sag_options = {'maxiter':max_iter, 'batch_size': batch_size, 'print_freq': 10}
fg_options = {'maxiter':max_iter, 'print_freq': 10}
lbfgsb_options = {'maxiter': max_iter, 'disp': False}
sg_options = {'maxiter':max_iter, 'batch_size': batch_size, 'print_freq': 10, 'step0': 1e-4, 'gamma': 0.55}

optimizer_options = [sag_options, fg_options, lbfgsb_options, sg_options]

run_methods(x_tr, y_tr, x_test, y_test, model_params, optimizer_options, file_name, ind_inputs_num, title, 'r2', True)