import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file

from vi_experiments import run_methods
from GP.covariance_functions import SquaredExponential

model_params = np.array([1.5, 1.5, 0.2])
model_covariance_obj = SquaredExponential(model_params)
ind_inputs_num = 50
max_iter = 100
file_name = 'small_real'

x_tr, y_tr = load_svmlight_file('../../../../Programming/DataSets/Regression/bodyfat(252, 14).txt')
data_name = 'bodyfat'
title = 'bodyfat dataset, n = 201, d = 14, m=50'

x_tr = x_tr.T
x_tr = x_tr.toarray()
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)
y_tr = scaler.fit_transform(y_tr)

x_tr = (x_tr + 1) / 2
y_tr = y_tr.reshape((y_tr.size, 1))
x_test = x_tr[:, int(x_tr.shape[1] * 0.8):]
y_test = y_tr[int(x_tr.shape[1] * 0.8):, :]
y_tr = y_tr[:int(x_tr.shape[1] * 0.8), :]
x_tr = x_tr[:, : int(x_tr.shape[1] * 0.8)]

lbfgsb_options = {'maxiter': max_iter, 'disp': False}
proj_newton_options = {'maxiter': max_iter, 'print_freq': 1}

optimizer_options = [lbfgsb_options, proj_newton_options]

run_methods(x_tr, y_tr, x_test, y_test, model_params, optimizer_options, file_name, ind_inputs_num, title, False)