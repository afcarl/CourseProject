import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file

from vi_experiments import run_methods
from GP.covariance_functions import SquaredExponential

file_name = 'huge_real'
model_params = np.array([1.0, 0.5, 0.5])
model_covariance_obj = SquaredExponential(model_params)
ind_inputs_num = 1000
max_iter = 10
batch_size = 500

x_tr, y_tr = load_svmlight_file('../../../../Programming/DataSets/Regression/yearprediction(463715, 90).txt')
data_name = 'cadata'
title = 'YearPredictionMSD dataset, n = 50000, d = 90, m=1000'
print('Read file')

x_tr = x_tr.T
x_tr = x_tr.toarray()
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)
y_tr = scaler.fit_transform(y_tr)
print('Performed scaling')

num = 50000
test_num = int(num / 10)
x_tr = (x_tr + 1) / 2
y_tr = y_tr.reshape((y_tr.size, 1))
x_test = x_tr[:, num:num+test_num]
y_test = y_tr[num:num+test_num, :]
y_tr = y_tr[:num, :]
x_tr = x_tr[:, :num]
print('formed training set')

lbfgsb_options = {'maxiter': max_iter, 'disp': False, 'mydisp': True}
proj_newton_options = {'maxiter': max_iter, 'print_freq': 1}

optimizer_options = [lbfgsb_options, proj_newton_options]

run_methods(x_tr, y_tr, x_test, y_test, model_params, optimizer_options, file_name, ind_inputs_num, title, True)