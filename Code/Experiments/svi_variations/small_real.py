import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file

from experiments_svi_variations import run_methods
from GP.covariance_functions import SquaredExponential

model_params = np.array([1.5, 1.5, 1.0])
model_covariance_obj = SquaredExponential(model_params)
ind_inputs_num = 100
max_iter = 200
batch_size = 100
file_name = 'small_real'
metric = 'loss'

if metric == 'loss':
    file_name += 'loss'
file_name += '.tikz'

x_tr, y_tr = load_svmlight_file('../../../../Programming/DataSets/Regression/bodyfat(252, 14).txt')
data_name = 'bodyfat'
title='bodyfat dataset, n = 201, d = 14, m=10'

x_tr = x_tr.T
x_tr = x_tr.toarray()
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)


x_tr = (x_tr + 1) / 2
y_tr = y_tr.reshape((y_tr.size, 1))
x_test = x_tr[:, int(x_tr.shape[1] * 0.8):]
y_test = y_tr[int(x_tr.shape[1] * 0.8):, :]
y_tr = y_tr[:int(x_tr.shape[1] * 0.8), :]
x_tr = x_tr[:, : int(x_tr.shape[1] * 0.8)]

# Cholesky parametrization

sag_options = {'maxiter':max_iter, 'batch_size': batch_size, 'print_freq': 100}
fg_options = {'maxiter':max_iter, 'print_freq': 100}
lbfgsb_options = {'maxiter': max_iter, 'disp': False}
sg_options = {'maxiter':max_iter, 'batch_size': batch_size, 'print_freq': 100, 'step0': 1e-4, 'gamma': 0.55}

optimizer_options = [sag_options, fg_options, lbfgsb_options, sg_options]

run_methods(x_tr, y_tr, x_test, y_test, model_params, optimizer_options, file_name, ind_inputs_num, title, metric, True)